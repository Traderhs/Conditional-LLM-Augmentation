# Qualitative Synthetic-Data Audit Findings (public release)

This redistribution-safe file records the descriptive observations from the deterministic 50-pair audit without redistributing third-party source text.

The original source text can be reconstructed locally from the cited dataset using the row identifier, source-row index, and SHA-256 text hash supplied in the CSV files.

## Representative examples

### English

- Row ID: `c42d8a077ea9260f0e72ab8e`
- Source-row index: `99`
- Source-text SHA-256: `60b5ed60c9296193769d997a012f2e1a512f1e97e28636fad7c5df46d2d9a0a7`
- Label: `0` / inherited label: `negative`
- Source text: omitted from the public package; retrieve it from the original dataset.
- Synthetic: The execution of the play was completely botched.
- Observation: Negative polarity is preserved; the very short source is expanded into a complete sentence with event context not explicit in the seed.

### Korean

- Row ID: `c0600632979a1f6d1cd9f8d1`
- Source-row index: `23956`
- Source-text SHA-256: `17f0fc7250061cc0a46c8e9cc72874d84b5ef0865a7e4a3de4efef75648de88b`
- Label: `1` / inherited label: `positive`
- Source text: omitted from the public package; retrieve it from the original dataset.
- Synthetic: 곡의 멜로디와 분위기가 정말 마음에 들었습니다.
- Observation: Positive evaluation is preserved; the short colloquial source is expanded from liking the song to explicitly praising its melody and atmosphere.

### Bengali

- Row ID: `b53e0079c2897601ec4c9a6a`
- Source-row index: `780`
- Source-text SHA-256: `1a0a920e43963421ba57cbd6199e630228826e6203dd63d454b9a3114ea66a16`
- Label: `1` / inherited label: `positive`
- Source text: omitted from the public package; retrieve it from the original dataset.
- Synthetic: চলচ্চিত্রটি এতটাই আবেগপ্রবণ ছিল যে আমি অশ্রুসিক্ত হলাম
- Observation: Positive affect and the reaction of being moved to tears are closely preserved in a more standardized sentence.

### Hausa

- Row ID: `bc3ef9f80c65eb04bc02e7a8`
- Source-row index: `423`
- Source-text SHA-256: `4e81f78885c83a26eaec2736007e81b24adc467c5cb38bd5ada085429a31a255`
- Label: `0` / inherited label: `negative`
- Source text: omitted from the public package; retrieve it from the original dataset.
- Synthetic: Wani labari ne mai ban tsoro da kuma rashin kunya, don haka ba zan taba sake duba shi ba
- Observation: Negative stance is preserved, but a specific conditional narrative complaint is generalized into a broader negative evaluation.

### Malayalam

- Row ID: `c60ffe24cfcfdece22b68636`
- Source-row index: `1766`
- Source-text SHA-256: `0f6e7a283c53be8262df0f77e70b2a3decf9fd24e98332ca51fbe939ef9347e7`
- Label: `0` / inherited label: `negative`
- Source text: omitted from the public package; retrieve it from the original dataset.
- Synthetic: വെറുതെ സമയം കളയാതെ, ഒരു സിനിമയെങ്കിലും വിജയിപ്പിക്കാൻ നോക്കൂ, അല്ലെങ്കിൽ ഈ മേഖലയിൽ നിന്നും വിട്ടുപോവുക.
- Observation: The negative evaluation of the film-related behavior is preserved; a Romanized Malayalam source is normalized into Malayalam script, while the synthetic text adds a stronger directive not stated explicitly in the source.

## Interpretation boundary

The audit is descriptive rather than a formal multilingual human-rating study. It contains no numerical human quality scores or inter-rater agreement statistics, and the selected examples were not chosen using downstream performance.
