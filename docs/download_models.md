# Download Pretrained Models

This guide explains how to manually download the required pretrained models using `wget`.

## Required Models

1. **bert-base-cased** - Standard BERT model
2. **Bio_ClinicalBERT** - Domain-specific BERT for medical/clinical text

---

## Download Commands

### 1. bert-base-cased

```bash
mkdir -p ./pretrained_models/bert-base-cased
cd ./pretrained_models/bert-base-cased

wget https://huggingface.co/bert-base-cased/resolve/main/config.json
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json
wget https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt

cd ../..
```

### 2. Bio_ClinicalBERT

```bash
mkdir -p ./pretrained_models/Bio_ClinicalBERT
cd ./pretrained_models/Bio_ClinicalBERT

wget https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json
wget https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/pytorch_model.bin
wget https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/tokenizer_config.json
wget https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/vocab.txt

cd ../..
```

---

## One-Liner Script

Copy and run this entire block:

```bash
# Create directories
mkdir -p ./pretrained_models/bert-base-cased
mkdir -p ./pretrained_models/Bio_ClinicalBERT

# Download bert-base-cased
wget -P ./pretrained_models/bert-base-cased https://huggingface.co/bert-base-cased/resolve/main/config.json
wget -P ./pretrained_models/bert-base-cased https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin
wget -P ./pretrained_models/bert-base-cased https://huggingface.co/bert-base-cased/resolve/main/tokenizer.json
wget -P ./pretrained_models/bert-base-cased https://huggingface.co/bert-base-cased/resolve/main/tokenizer_config.json
wget -P ./pretrained_models/bert-base-cased https://huggingface.co/bert-base-cased/resolve/main/vocab.txt

# Download Bio_ClinicalBERT
wget -P ./pretrained_models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/config.json
wget -P ./pretrained_models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/pytorch_model.bin
wget -P ./pretrained_models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/tokenizer_config.json
wget -P ./pretrained_models/Bio_ClinicalBERT https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT/resolve/main/vocab.txt
```

---

## Expected Directory Structure

After downloading, your directory should look like this:

```
pretrained_models/
├── bert-base-cased/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── Bio_ClinicalBERT/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    └── vocab.txt
```

---

## Verify Downloads

Check file sizes to ensure downloads completed:

```bash
ls -lh ./pretrained_models/bert-base-cased/
ls -lh ./pretrained_models/Bio_ClinicalBERT/
```

Expected sizes (approximate):
- `pytorch_model.bin`: ~420 MB (bert-base-cased) / ~420 MB (Bio_ClinicalBERT)
- `config.json`: ~1 KB
- `vocab.txt`: ~200 KB
- `tokenizer.json`: ~400 KB (bert-base-cased only)
- `tokenizer_config.json`: ~1 KB

---

## Troubleshooting

### SSL Certificate Error
If you get SSL errors, add `--no-check-certificate`:
```bash
wget --no-check-certificate https://huggingface.co/...
```

### Slow Download
Use `-c` to resume interrupted downloads:
```bash
wget -c https://huggingface.co/...
```

### Alternative: curl
If `wget` is not available, use `curl`:
```bash
curl -L -o ./pretrained_models/bert-base-cased/config.json https://huggingface.co/bert-base-cased/resolve/main/config.json
```
