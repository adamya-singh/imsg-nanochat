# Training Quality Review v1

Review each two-message example only for usefulness in reply-style training.

## Include

- Natural inbound text followed by a coherent human reply.
- Short replies, slang, profanity, unusual style, personal content, and context-dependent texting when they still represent real reply behavior.
- Minor typos or grammatical mistakes that reflect authentic texting style.

## Exclude

- Extraction/archive artifacts, serialization metadata, or corrupted text.
- Automated service, spam, OTP, notification, or transactional messages.
- Empty or malformed pairs.
- Replies that are clearly incoherent, unrelated to the inbound message, or not representative of a reply interaction.
- Exact or near-exact duplicates that add no distinct training signal.

## Unsure

Use `unsure` when context is insufficient and neither inclusion nor exclusion is clearly justified. Bias toward inclusion rather than excluding natural conversation.

Do not exclude solely because content is private, sensitive, profane, unusual, or personally identifying.
