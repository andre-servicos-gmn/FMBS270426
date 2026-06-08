# ESCOPO — Base Sports Agent

> Agente conversacional WhatsApp para franquia de Beach Tennis e Padel.
> Stack: FastAPI + LangGraph + Supabase (pgvector) + Redis + Evolution API + OpenAI.

---

## Estado atual (baseline)

O projeto já tem base sólida funcionando no CLI:

- Grafo LangGraph com 6 nodes (triage, diagnose, recommend, close, faq, handoff, smalltalk) e roteamento inteligente
- Slot-filling LLM-driven (adaptativo, não-linear) com auto-preenchimento para iniciantes
- RAG vetorial real no Supabase com guardrail anti-alucinação de produto
- 20 produtos seedados, conversa funcional end-to-end no CLI
- Mascaramento de PII obrigatório antes de cada chamada LLM
- Idempotência de mensagens via Redis

**Gaps que este escopo resolve:**

1. Slot `regiao_lesao` e `modelo_desejado` não existem (cliente pediu explicitamente)
2. Persistência de conversa perde tudo em restart (MemorySaver in-memory)
3. Schema de produtos é racket-only — não cobre vestuário, bolas, acessórios
4. Não processa foto nem áudio do cliente
5. Evolution API não está conectada à instância real
6. Stubs órfãos no código (`anthropic_client.py`, `RedisSessionStore`)

---

## Sprint 0 — Limpeza (½ dia)

**Objetivo:** zerar ruído do código antes de adicionar features novas.

### Tarefas

1. **Deletar `app/adapters/anthropic_client.py`**
   Decisão tomada: projeto é OpenAI-only. Stub não tem motivo de existir.

2. **Decidir destino do `RedisSessionStore` (`app/storage/redis_session.py`)**
   Classe órfã, não importada. Como na Sprint 1 vamos usar `langgraph-checkpoint-redis` (que tem sua própria abstração), o `RedisSessionStore` provavelmente fica obsoleto. **Deletar**, a menos que tenha uso em idempotência hoje.

3. **Desligar APScheduler do catalog sync enquanto `CATALOG_API_URL` estiver vazio**
   Hoje roda a cada 6h batendo em URL vazia, polui logs. Condicionar inicialização à presença da env var.

4. **Atualizar `CLAUDE.md` do projeto**
   Remover menção a Anthropic, refletir stack real (OpenAI), e documentar próximos passos deste escopo.

### Critério de aceitação
- Logs limpos em startup local (sem warnings de catalog sync ou Anthropic)
- `grep -r "anthropic" app/` retorna apenas dependências legítimas (não retorna nada se for o caso)
- Testes existentes continuam passando

---

## Sprint 1 — MVP WhatsApp (2–3 dias)

**Objetivo:** agente respondendo no WhatsApp real, com persistência sólida e slots completos.

### 1.1 — Migrar MemorySaver → RedisSaver

**Arquivo:** `app/agent/graph.py`

- Instalar `langgraph-checkpoint-redis` (verificar versão estável atual)
- Substituir `MemorySaver()` por `RedisSaver` apontando para o Redis Cloud já configurado
- Garantir que `thread_id=phone_hash` continua sendo a chave de continuidade
- TTL nos checkpoints: 7 dias (depois disso conversa "esfria" e cliente começa do zero)

**Critério:** reiniciar servidor no meio de uma conversa no CLI e ela continuar de onde parou.

### 1.2 — Adicionar slots `regiao_lesao` e `modelo_desejado`

**Arquivos:** `app/agent/state.py`, `app/agent/prompts.py`, `app/agent/nodes/diagnose.py`

**No `AgentState.player_profile`, adicionar:**

```python
regiao_lesao: Optional[str]      # cotovelo, ombro, punho, antebraço, mais de uma, nenhuma
modelo_desejado: Optional[str]    # nome/marca que o cliente já tem em mente, ou "nenhum"
```

**No `SYSTEM_DIAGNOSE`, adicionar regras:**

```
E. Se o cliente indicar QUALQUER lesão ou dor (lesoes != "nenhuma"),
   pergunte EM SEGUIDA: "Em qual região? (cotovelo, ombro, punho, antebraço, 
   braço inteiro, ou mais de uma)". Registre em regiao_lesao.
   Se lesoes == "nenhuma", preencha regiao_lesao = "nenhuma" automaticamente.

F. Após coletar perfil técnico básico, pergunte UMA VEZ:
   "Você já tem algum modelo ou marca em mente?"
   - Se sim, registre em modelo_desejado
   - Se "não sei" / "quero indicação" / similar, registre "nenhum"
   NÃO insista nem ofereça opções nesta pergunta — é apenas captura.
```

**Em `recommend.py`:**
- `_build_filters` deve usar `regiao_lesao` para priorizar produtos com `antivibração` / `flexibilidade alta` (campo no `attributes` quando Sprint 2 estiver pronta — por enquanto, injetar como contexto textual no prompt do recommend)
- `modelo_desejado` entra como dica no prompt do recommend: se o catálogo tiver match exato ou similar, priorizar; se não tiver, mencionar e sugerir alternativa

**Critério:** conversa no CLI coleta os 2 slots novos quando aplicável e a recomendação faz referência a eles ("como você mencionou dor no cotovelo, priorizei modelos com…").

### 1.3 — Conectar Evolution API real

**Arquivos:** `.env`, painel Evolution

- Preencher `.env`: `EVOLUTION_API_URL`, `EVOLUTION_API_KEY`, `EVOLUTION_INSTANCE`, `EVOLUTION_WEBHOOK_TOKEN`
- Subir ngrok apontando para `localhost:8000`
- No painel da Evolution, configurar webhook para `https://<ngrok>/webhook/whatsapp` com header `apikey`
- Habilitar evento `messages.upsert` (e `messages.update` se quisermos tratar edits — opcional)

**Critério:** mandar mensagem real no WhatsApp e receber resposta do agente. Conversa salva em `conversation_logs` e `leads`.

### 1.4 — CTA da loja com dados reais

**Arquivos:** `.env`, `app/agent/prompts.py` (SYSTEM_CLOSE)

- Adicionar `.env`: `STORE_NAME`, `STORE_ADDRESS`, `STORE_HOURS`, `STORE_MAPS_URL`, `STORE_PHONE`
- Injetar essas variáveis no contexto do `close_node` para que a mensagem final tenha endereço, horário e link do Maps

**Critério:** mensagem de fechamento inclui nome da unidade, endereço, horário e link do Maps clicável.

---

## Sprint 2 — Catálogo flexível (2 dias)

**Objetivo:** suportar todos os produtos da loja com schema extensível.

### 2.1 — Migration do schema

**Adicionar à tabela `products`:**

```sql
ALTER TABLE products 
  ADD COLUMN category TEXT NOT NULL DEFAULT 'raquete'
    CHECK (category IN ('raquete', 'bola', 'vestuario', 'acessorio', 'calcado', 'bolsa')),
  ADD COLUMN attributes JSONB DEFAULT '{}'::jsonb;

-- Migrar dados existentes:
UPDATE products 
SET attributes = jsonb_build_object(
  'weight_g', weight_g,
  'balance', balance,
  'material', material,
  'level', level
) WHERE category = 'raquete';

-- Manter colunas antigas por compatibilidade até refatorar tudo, 
-- ou dropar e atualizar search_products() em uma migration só (recomendo).

CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_attributes ON products USING gin(attributes);
```

**Versionar com Alembic** (não existe ainda — criar setup mínimo nessa sprint).

### 2.2 — Atualizar `search_products()` SQL function

- Aceitar parâmetro `category` opcional
- Permitir filtros em `attributes` via JSONB (ex: `attributes->>'tem_antivibracao' = 'true'`)
- Manter cosine similarity em embedding como antes

### 2.3 — Refatorar `recommend.py`

**Arquivo:** `app/agent/nodes/recommend.py`

- `_build_query` parametrizado por `category` (não mais hardcoded "raquete")
- `_build_filters` lê do `player_profile` e mapeia para filtros + atributos:
  - `regiao_lesao` → busca produtos com `attributes->>'flexivel' = 'true'`
  - `nivel_jogo` → mapeia para `attributes->>'level'`
  - `orcamento` → `price_cents <= X`
- Categoria default: inferir do `intent`/`messages` recentes. Se cliente disse "quero uma raquete" → `category=raquete`. Se "tem bolsa térmica?" → `category=acessorio`.

### 2.4 — Detector de categoria no triage

**Arquivo:** `app/agent/nodes/triage.py` ou node novo

- Quando intent = `diagnose` ou `recommend`, classificar também a `category` desejada
- Adicionar `category_interest` ao `AgentState`
- Se for `raquete` → fluxo completo de diagnose (perfil técnico)
- Se for `vestuario`/`acessorio`/`bola` → fluxo curto (só tamanho/cor/finalidade conforme categoria)

### 2.5 — Reseed do catálogo com variedade

**Arquivo:** `scripts/seed_via_rest.py`

- Adicionar pelo menos: 8 raquetes (com `attributes` completos), 4 bolas, 4 peças de vestuário, 4 acessórios
- Cada produto com `description` rica para o embedding capturar bem

### Critério de aceitação Sprint 2
- Conversa: "quero uma bermuda" → agente coleta tamanho/cor → recomenda da categoria certa
- Conversa: "tem raquete?" → fluxo completo de diagnose como hoje
- Migration reversível, dados antigos preservados

---

## Sprint 3 — Mídia (foto + áudio) (2 dias)

**Objetivo:** aceitar foto e áudio do cliente como input.

### 3.1 — Handler de mídia no webhook

**Arquivo:** `app/api/webhook.py`

Atualmente `_extract_text()` só lê `message.conversation` e `message.extendedTextMessage.text`. Estender para detectar:

- `message.imageMessage` → baixar imagem via Evolution
- `message.audioMessage` → baixar áudio via Evolution
- `message.documentMessage` → por ora, ignorar com mensagem amigável ("manda como foto ou texto que te ajudo melhor")
- `message.stickerMessage` → ignorar silenciosamente

### 3.2 — Cliente Evolution para download de mídia

**Arquivo:** `app/adapters/evolution.py`

- Adicionar método `get_media_base64(message_id, instance)` chamando `POST /chat/getBase64FromMediaMessage/<instance>`
- Retornar bytes + mime_type

### 3.3 — Processadores de mídia

**Novo arquivo:** `app/adapters/media_processor.py`

- `transcribe_audio(audio_bytes, mime_type) -> str` usando OpenAI Whisper (`whisper-1`)
- `describe_image(image_bytes, mime_type, context: str) -> str` usando GPT-4o vision
  - Context pode ser: "o cliente está procurando raquete e mandou esta foto — descreva a raquete (marca, modelo se visível, características)"

### 3.4 — Injeção como HumanMessage

**Arquivo:** `app/api/webhook.py`

Após processar mídia, criar `HumanMessage` com prefixo identificando origem:

```
[ÁUDIO TRANSCRITO]: <texto do whisper>
[IMAGEM RECEBIDA — descrição]: <descrição do GPT-4o>
```

E passar pro grafo como qualquer outra mensagem. O LLM trata como texto a partir daí.

### 3.5 — Atualizar prompt do diagnose

Adicionar regra no `SYSTEM_DIAGNOSE`:

```
G. Quando receber [IMAGEM RECEBIDA] descrevendo uma raquete, registre o que 
   conseguir identificar em equipamento_atual (se for raquete do cliente) ou 
   em modelo_desejado (se cliente disse algo como "quero uma assim").
   Confirme com o cliente: "É essa aqui da foto que você usa hoje, certo?"
```

### Critério de aceitação Sprint 3
- Cliente manda áudio "quero uma raquete intermediária" → agente transcreve e processa normalmente
- Cliente manda foto de raquete antiga → agente descreve, registra em `equipamento_atual`, confirma
- Mídia não suportada (documento) → resposta amigável sem quebrar o fluxo

---

## Resumo de priorização

| Sprint | Duração | Bloqueia o quê? |
|--------|---------|-----------------|
| 0 — Limpeza | ½ dia | Nada, mas reduz ruído |
| 1 — MVP WhatsApp | 2–3 dias | Tudo. Sem isso não tem produto real |
| 2 — Catálogo flexível | 2 dias | Atender cliente que pede "outros produtos" |
| 3 — Mídia | 2 dias | Diferencial competitivo, mas não bloqueante |

**Total estimado:** ~7 dias úteis de execução com Claude Code.

---

## Sprint 2.6 — Diagnose deprecated no WhatsApp (refatoração estratégica)

Em produção (Maio/2026), o `diagnose` ativo no agente do WhatsApp estava canibalizando o valor da Consultoria Base Sports (R$350, abatido na compra). O classificador frequentemente roteava clientes determinados pra diagnose, forçando perguntas sobre nível/lesão/esporte prévio que (a) atrasavam quem vinha decidido, e (b) entregavam de graça parte do diagnóstico que é o produto principal.

**Decisão estratégica:** o agente do WhatsApp NUNCA faz diagnóstico longo. Atende, vende, convida pra loja. Quando o cliente pede orientação genérica (sem nomear produto), o agente oferece:
1. **Consultoria com especialista** — R$350, abatido se fechar raquete no mesmo dia.
2. **Visita à loja** — gratuita, presencial.

**Mudanças técnicas (Sprint 2.6):**
- `diagnose` removido do grafo LangGraph (arquivo mantido com docstring DEPRECATED para futura Consultoria virtual).
- Triage simplificado pra 9 intents declarativos (`smalltalk`, `product_inquiry`, `price_inquiry`, `purchase_intent`, `scheduling_inquiry`, `out_of_scope`, `faq`, `help_request`, `close`).
- State limpo: `customer_intent_path` e `awaiting_alternatives_decision` removidos (não fazem mais sentido sem o fork determined/exploring).
- `recommend.py` simplificado — sem REFERENCE-SIM / REFERENCE-NÃO / PROFILE; única responsabilidade: ver se o produto existe no Bling e responder com confirmação curta.
- Novo nó `help_request` com 3 variações deterministicamente pickadas por `phone_hash`.
- Tabela legacy `products` esvaziada via migration `0008_truncate_products_seed.sql` (agente lê de `bling_products` quando Bling ativo).
- ~95 testes da era diagnose marcados como skipped com motivo "diagnose deprecated in Sprint 2.6".
- 15 testes novos em `tests/test_help_request.py`.

## Decisões registradas

- **OpenAI permanece**, Anthropic descartado (stub removido)
- **Schema flexível com `category` + `attributes JSONB`** (Opção A)
- **RedisSaver antes do WhatsApp** (não aceitar perda de contexto em prod)
- **Foto + áudio no MVP** (Whisper + GPT-4o vision)
- **CTA fecha em loja física** com dados reais (endereço/horário/Maps)

## Itens fora deste escopo (backlog)

- Rate limiting na webhook
- Testes automatizados rodando em CI
- Tratar `messages.update` (edição de mensagens)
- Handoff humano via sinalização para atendente (hoje só marca lead)
- Métricas/analytics (quantos leads, taxa de conversão para visita à loja)