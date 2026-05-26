# Setup Evolution API + WhatsApp end-to-end

Guia passo a passo para conectar o agente local a uma instância da Evolution API
e validar uma mensagem real fluindo do WhatsApp até a resposta do agente.

Pré-requisitos:
- Servidor FastAPI rodando localmente em `localhost:8000`
- Conta na Evolution API (auto-hospedada ou em provedor gerenciado)
- WhatsApp já pareado com a instância (QR code lido)
- ngrok 3.20+ instalado e autenticado (`ngrok config add-authtoken <seu_token>`)

## 1. Obter credenciais da Evolution

No painel da sua instância da Evolution, você precisa de quatro valores:

| Variável de ambiente   | O que é                                                                                  |
|------------------------|------------------------------------------------------------------------------------------|
| `EVOLUTION_API_URL`    | Base URL da instância — ex. `https://evo.minhafranquia.com` (sem `/` no final)            |
| `EVOLUTION_API_KEY`    | Chave global da Evolution (header `apikey` nas chamadas de envio)                         |
| `EVOLUTION_INSTANCE`   | Nome da instância configurada no painel — aparece em `/instance/fetchInstances`           |
| `EVOLUTION_WEBHOOK_TOKEN` | Token escolhido por você (qualquer string secreta). A Evolution vai enviá-lo no header `apikey` em cada webhook que postar para o nosso servidor. Use o mesmo valor quando configurar o webhook no painel — o nosso `/webhook/whatsapp` vai comparar contra essa variável. **Se a sua build da Evolution não permite configurar headers customizados no webhook, deixe esta variável vazia no `.env`** — a autenticação será desabilitada, um WARNING estruturado é logado em cada request, e o webhook aceita qualquer chamada. Reabilite antes de produção. |

Cole esses valores no `.env` na raiz do projeto. O `.env.example` lista os nomes
exatos. Depois reinicie o `uvicorn` para o pydantic-settings recarregar.

## 2. Subir o servidor FastAPI

```bash
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

Esperado no log:
- `redis_checkpointer initialized ttl_minutes=10080` (Sprint 1.1)
- `catalog sync disabled (no CATALOG_API_URL)` se a sync ainda não estiver configurada (Sprint 0)
- `Application startup complete.`

Teste rapidamente: `curl http://localhost:8000/health` deve retornar
`{"status":"ok","env":"development"}`.

## 3. Subir o túnel ngrok

A Evolution API roda fora do seu computador e precisa de uma URL pública para
postar webhooks. Use ngrok:

```bash
ngrok http 8000
```

A saída vai mostrar uma URL parecida com:

```
Forwarding   https://chevron-lustily-edition.ngrok-free.dev -> http://localhost:8000
```

A URL muda cada vez que o ngrok reinicia (a menos que você assine o plano pago
com domínio fixo). Para o teste de hoje, copie a URL atual.

Confira que a URL pública chega no FastAPI:

```bash
curl -H "ngrok-skip-browser-warning: 1" \
     https://chevron-lustily-edition.ngrok-free.dev/health
```

Resposta esperada: `{"status":"ok","env":"development"}`.

## 4. Configurar webhook no painel da Evolution

No painel da Evolution, vá em **Instâncias → sua instância → Webhook** (o caminho
exato varia por versão; em algumas builds é chamado de "Events" ou "Settings").

Preencha:

- **URL**: `https://<seu-subdominio>.ngrok-free.dev/webhook/whatsapp`
- **Header (apikey)**: o mesmo valor de `EVOLUTION_WEBHOOK_TOKEN` no seu `.env`
- **Eventos a habilitar**: no mínimo `MESSAGES_UPSERT` (recebe mensagens novas).
  Opcionalmente `MESSAGES_UPDATE` (edits) — hoje o nosso webhook ignora esse
  evento, mas habilitar não atrapalha.

> Se sua versão da Evolution não tiver UI para webhook, use a chamada REST:
>
> ```
> POST {EVOLUTION_API_URL}/webhook/set/{EVOLUTION_INSTANCE}
> Header: apikey: {EVOLUTION_API_KEY}
> Body: {
>   "url": "https://<seu-subdominio>.ngrok-free.dev/webhook/whatsapp",
>   "webhook_by_events": false,
>   "webhook_base64": false,
>   "events": ["MESSAGES_UPSERT"]
> }
> ```

## 5. Testar end-to-end

Mande uma mensagem qualquer pelo WhatsApp para o número que está pareado com
a instância. O fluxo é:

1. WhatsApp → Evolution API
2. Evolution → `POST https://<ngrok>/webhook/whatsapp` (com header `apikey`)
3. FastAPI valida o token, normaliza o payload, despacha em BackgroundTask
4. LangGraph processa, salva checkpoint no Redis
5. Resposta sai via `EvolutionClient.send_text()` → Evolution → WhatsApp

Observe no log do uvicorn:
- `triage intent=<x>`
- `evolution_send_text begin phone=<8 chars> instance=<...> text_len=<n>`
- `evolution_send_text phone=<8 chars> status=200 attempt=1`

Se a resposta não chegar, checar em ordem:
- `curl /health` na URL ngrok funciona? Se não → ngrok caiu ou FastAPI caiu
- Log mostra `Unauthorized`? → `EVOLUTION_WEBHOOK_TOKEN` no `.env` difere do
  apikey configurado no painel
- Log mostra `evolution_send_text failed`? → credenciais de envio (`EVOLUTION_API_URL`,
  `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE`) erradas ou a Evolution está offline
- Webhook recebe mas grafo não responde? → checar log do Redis (checkpoint
  ou idempotência indisponível)

## 6. Diagnóstico rápido pelo painel do ngrok

`http://localhost:4040` abre o inspector do ngrok. Lá você vê cada request
recebido em tempo real, com headers e corpo — útil para confirmar exatamente
o que a Evolution está mandando e o que o FastAPI está respondendo. Mensagens
duplicadas aparecem como 200 com `{"status":"ok","duplicate":true}` (idempotência
via Redis funcionou).
