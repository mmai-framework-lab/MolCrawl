# GPT-2 Model Inference Feature

## Overview

This feature adds interactive inference capabilities to the GPT-2 Training Status web interface. Users can click on any trained model card to open a modal dialog that displays model details and allows real-time text generation.

## Components

### 1. Backend API (`api/gpt2-inference.js`)

**Endpoints:**

- `POST /api/gpt2-inference` - Run inference on a GPT-2 model
  - **Request Body:**

    ```json
    {
      "dataset": "compounds|genome_sequence|protein_sequence|rna|molecule_nl",
      "size": "small|medium|large|xl",
      "prompt": "Input text prompt",
      "maxLength": 128,
      "temperature": 1.0,
      "topK": 50,
      "numSamples": 1
    }
    ```

  - **Response:**

    ```json
    {
      "success": true,
      "results": ["generated text 1", "generated text 2"]
    }
    ```

- `GET /api/gpt2-inference/config/:dataset` - Get default inference configuration
  - **Response:**

    ```json
    {
      "success": true,
      "config": {
        "name": "Compounds",
        "start_token": "<|startofsmiles|>",
        "max_length": 128,
        "temperature": 1.0,
        "top_k": null,
        "example_prompts": ["<|startofsmiles|>C", "..."]
      }
    }
    ```

**Features:**

- Supports both HuggingFace and legacy checkpoint formats
- Automatically locates the latest checkpoint for each model
- Dataset-specific default configurations
- Python-based inference execution using the existing GPT model code

### 2. Frontend Modal Component (`src/InferenceModal.js`)

**Props:**

- `isOpen` (boolean) - Controls modal visibility
- `onClose` (function) - Callback when modal is closed
- `dataset` (string) - Dataset identifier
- `size` (string) - Model size (small, medium, large, xl)
- `modelData` (object) - Model metadata and checkpoint information

**Features:**

- **Model Information Display:**
  - Training iteration count
  - Validation loss
  - Model parameters (layers, embedding dimension, etc.)
  - Current status

- **Interactive Inference Controls:**
  - Text prompt input with example suggestions
  - Adjustable generation parameters:
    - Max Length (32-512 tokens)
    - Temperature (0.1-2.0)
    - Top-K sampling (0-200)
    - Number of samples (1-10)
- **Results Display:**
  - Multiple generated samples
  - Copy-to-clipboard functionality
  - Formatted output with syntax highlighting

### 3. Updated Training Status Component (`src/GPT2TrainingStatus.js`)

**New Features:**

- Clickable model cards with hover effects
- Modal state management
- Automatic model data passing to inference modal

**Visual Indicators:**

- Clickable cards have a special hover effect with purple border
- Cursor changes to pointer on ready models
- Non-trained models remain non-clickable

### 4. Styling (`src/InferenceModal.css`)

**Design Features:**

- Modern gradient header with purple theme
- Responsive layout (mobile-friendly)
- Smooth animations and transitions
- Custom styled range inputs
- Scrollable content areas
- Professional color scheme matching the existing interface

## Usage

### For Users

1. Navigate to the GPT-2 Training Status page
2. Look for model cards with checkpoint data (shows training statistics)
3. Click on any trained model card
4. The inference modal will open with:
   - Model details at the top
   - Generation parameters in the middle
   - A "Generate" button to run inference
5. Adjust parameters as needed:
   - Use example prompts for quick testing
   - Adjust temperature for creativity (lower = more deterministic)
   - Set top-K for sampling diversity
   - Choose number of samples to generate
6. Click "Generate" to run inference
7. View results and copy them to clipboard as needed

### Dataset-Specific Configurations

| Dataset          | Start Token | Max Length    | Temperature | Top-K | Use Case |
| ---------------- | ----------- | ------------- | ----------- | ----- | -------- | ---- | ----------------------------- |
| Compounds        | `<          | startofsmiles | >`          | 128   | 1.0      | None | SMILES generation             |
| Genome Sequence  | `<          | startoftext   | >`          | 256   | 0.8      | 200  | DNA sequences                 |
| Protein Sequence | `<          | startoftext   | >`          | 256   | 0.8      | 50   | Amino acid sequences          |
| RNA              | `<          | startoftext   | >`          | 256   | 0.8      | 100  | RNA sequences                 |
| Molecule NL      | `<          | startoftext   | >`          | 128   | 0.7      | 50   | Natural language descriptions |

## Technical Details

### Checkpoint Detection

The system automatically detects checkpoints in two formats:

1. **HuggingFace Format** (preferred):
   - Located in `checkpoint-XXXX` directories
   - Uses HuggingFace transformers library
   - Format: `model.generate()` API

2. **Legacy Format**:
   - Located as `ckpt.pt` file
   - Uses custom GPT model class
   - Format: Custom `model.generate()` implementation

### Python Integration

The backend spawns a Python process to run inference:

- Dynamically generates Python script with parameters
- Loads the appropriate model and tokenizer
- Handles both checkpoint formats seamlessly
- Returns results as JSON

### Error Handling

- Model not found errors
- Checkpoint loading failures
- Inference execution errors
- Python environment issues
- All errors are displayed in the modal with helpful messages

## Installation & Setup

1. Ensure the backend dependencies are installed:

   ```bash
   cd molcrawl-web
   npm install
   ```

2. Ensure Python dependencies are available:

   ```bash
   pip install torch transformers
   ```

3. Set the `LEARNING_SOURCE_DIR` environment variable:

   ```bash
   export LEARNING_SOURCE_DIR="learning_source_202508"
   ```

4. Start the server:

   ```bash
   npm start
   ```

## Future Enhancements

Potential improvements for future versions:

1. **Batch Generation**: Support for generating multiple samples in parallel
2. **Model Comparison**: Side-by-side generation from different model sizes
3. **History**: Save and review previous generations
4. **Advanced Parameters**: Additional generation controls (top-p, repetition penalty)
5. **Export**: Download generated sequences in various formats
6. **Visualization**: Display attention weights or probability distributions
7. **API Keys**: Secure access control for production deployment
8. **Queue System**: Handle multiple concurrent inference requests
9. **Streaming**: Real-time token-by-token generation display
10. **Fine-tuning**: Quick fine-tuning interface for specific use cases

## Troubleshooting

### Modal doesn't open

- Check browser console for errors
- Ensure the model has a valid checkpoint
- Verify the dataset configuration exists

### Inference fails

- Check Python environment is properly configured
- Verify checkpoint files exist and are readable
- Check server logs for detailed error messages

### Slow generation

- GPU availability impacts speed significantly
- Larger models take longer to generate
- Consider reducing `maxLength` or `numSamples`

## Architecture Diagram

```
User clicks model card
       ↓
GPT2TrainingStatus.js
       ↓ (opens modal)
InferenceModal.js
       ↓ (POST request)
/api/gpt2-inference
       ↓ (spawns process)
Python Script
       ↓ (loads model)
gpt2/model.py
       ↓ (generates)
Results → InferenceModal → User
```

## Security Considerations

- Input sanitization for prompts
- Process timeout to prevent hanging
- Resource limits for generation
- Access control for production deployment
- Rate limiting for API endpoints

## Performance Notes

- First generation may be slow due to model loading
- Subsequent generations reuse loaded model (when using HF format)
- GPU acceleration significantly improves speed
- Memory usage scales with model size
