# GPT-2 Inference Feature - Visual Guide

## User Interface Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GPT-2 Training Status Dashboard                       │
│                                                                           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │   SMALL    │  │   MEDIUM   │  │   LARGE    │  │     XL     │       │
│  │  ✓ Ready   │  │  ✓ Ready   │  │ 🚀 Training│  │ Not Started│       │
│  │            │  │            │  │            │  │            │       │
│  │ Iter: 10K  │  │ Iter: 8K   │  │ Iter: 5K   │  │            │       │
│  │ Loss: 2.1  │  │ Loss: 1.9  │  │ Loss: 1.7  │  │            │       │
│  │            │  │            │  │            │  │            │       │
│  │[Clickable] │  │[Clickable] │  │[Clickable] │  │[Disabled]  │       │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
│       ↓ CLICK                                                            │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                     🧬 GPT-2 Inference Modal                     ✕      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  📊 Model Details                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Status: ✓ Ready         │ Iteration: 10,000                     │   │
│  │ Val Loss: 2.1234        │ Parameters: 85M                       │   │
│  │ Layers: 12              │ Embedding: 768                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  ⚙️ Generation Parameters                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Prompt:                                                          │   │
│  │ ┌──────────────────────────────────────────────────────────┐   │   │
│  │ │ <|startofsmiles|>C                                        │   │   │
│  │ └──────────────────────────────────────────────────────────┘   │   │
│  │ Examples: [<|startofsmiles|>C] [<|startofsmiles|>O]  ...    │   │   │
│  │                                                                  │   │
│  │ Max Length: 128        ◄─────────►                              │   │
│  │ Temperature: 1.00      ◄─────────►                              │   │
│  │ Top-K: 50              ◄─────────►                              │   │
│  │ Num Samples: 1         [1]                                      │   │
│  │                                                                  │   │
│  │              [🚀 Generate]                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  📝 Generated Results (3)                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ #1                                             [📋 Copy]         │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ CC(C)C(=O)Nc1ccc(O)cc1                                      │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │ #2                                             [📋 Copy]         │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ c1ccc(Br)cc1                                                │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│                                                   [Close]                │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           Browser                                     │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │            GPT2TrainingStatus.js (Main Component)              │ │
│  │                                                                 │ │
│  │  State:                                                         │ │
│  │  • trainingData      → Model status & metrics                 │ │
│  │  • modalOpen         → Modal visibility                        │ │
│  │  • selectedModel     → Currently selected model                │ │
│  │                                                                 │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │  Model Cards (Grid Layout)                               │ │ │
│  │  │                                                           │ │ │
│  │  │  [Small]  [Medium]  [Large]  [XL]                        │ │ │
│  │  │     ↓        ↓         ↓       ↓                          │ │ │
│  │  │  onClick  onClick  onClick  onClick                       │ │ │
│  │  │     ↓        ↓         ↓       ↓                          │ │ │
│  │  │  handleModelClick() → setModalOpen(true)                 │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  │                          ↓                                      │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │         InferenceModal.js (Modal Component)              │ │ │
│  │  │                                                           │ │ │
│  │  │  Props:                                                   │ │ │
│  │  │  • isOpen            → boolean                           │ │ │
│  │  │  • dataset           → "compounds", "rna", etc.          │ │ │
│  │  │  • size              → "small", "medium", etc.           │ │ │
│  │  │  • modelData         → checkpoint info                   │ │ │
│  │  │                                                           │ │ │
│  │  │  State:                                                   │ │ │
│  │  │  • prompt            → input text                        │ │ │
│  │  │  • maxLength         → generation length                 │ │ │
│  │  │  • temperature       → sampling temp                     │ │ │
│  │  │  • topK              → top-k sampling                    │ │ │
│  │  │  • results           → generated texts                   │ │ │
│  │  │                                                           │ │ │
│  │  │  ┌────────────────────────────────────────────────────┐ │ │ │
│  │  │  │  [Generate Button]                                 │ │ │ │
│  │  │  │       ↓                                            │ │ │ │
│  │  │  │  handleInference()                                 │ │ │ │
│  │  │  │       ↓                                            │ │ │ │
│  │  │  │  POST /api/gpt2-inference                          │ │ │ │
│  │  │  └────────────────────────────────────────────────────┘ │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ HTTP POST
┌──────────────────────────────────────────────────────────────────────┐
│                         Express Server                                │
│                                                                       │
│  server.js                                                           │
│  ├── Route: /api/gpt2-inference                                      │
│  │      ↓                                                            │
│  └── gpt2-inference.js                                               │
│       │                                                              │
│       ├─→ validateParams()                                           │
│       ├─→ getCheckpointPath(dataset, size)                           │
│       │    ├─→ Check for HuggingFace format                          │
│       │    └─→ Check for legacy format                               │
│       │                                                              │
│       └─→ runInference()                                             │
│            │                                                          │
│            ├─→ Generate Python script                                │
│            ├─→ spawn('python', ['-c', script])                       │
│            │                                                          │
│            ↓                                                          │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ Python Process
┌──────────────────────────────────────────────────────────────────────┐
│                       Python Inference                                │
│                                                                       │
│  1. Import libraries (torch, transformers)                           │
│  2. Load model:                                                       │
│     ├─→ HuggingFace: GPT2LMHeadModel.from_pretrained()              │
│     └─→ Legacy: torch.load(ckpt.pt)                                  │
│  3. Load tokenizer                                                    │
│  4. Encode prompt                                                     │
│  5. Generate:                                                         │
│     model.generate(                                                   │
│       input_ids,                                                      │
│       max_length=maxLength,                                          │
│       temperature=temperature,                                        │
│       top_k=topK                                                      │
│     )                                                                 │
│  6. Decode results                                                    │
│  7. Output JSON to stdout                                             │
│                                                                       │
│  Output: {"success": true, "results": ["text1", "text2"]}           │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ JSON Response
┌──────────────────────────────────────────────────────────────────────┐
│                         Back to Browser                               │
│                                                                       │
│  InferenceModal.js                                                   │
│  ├── Receive response                                                │
│  ├── setResults(response.results)                                    │
│  ├── Display results                                                 │
│  └── Enable copy-to-clipboard                                        │
└──────────────────────────────────────────────────────────────────────┘
```

## API Request/Response Flow

```
Frontend                     Backend                      Python
   │                           │                            │
   │  POST /api/gpt2-inference │                            │
   │  {                        │                            │
   │    dataset: "compounds",  │                            │
   │    size: "small",         │                            │
   │    prompt: "<|start|>C",  │                            │
   │    maxLength: 128,        │                            │
   │    temperature: 1.0       │                            │
   │  }                        │                            │
   ├──────────────────────────►│                            │
   │                           │  Validate params           │
   │                           │  Find checkpoint           │
   │                           │  Generate Python script    │
   │                           ├───────────────────────────►│
   │                           │                            │ Load model
   │                           │                            │ Load tokenizer
   │                           │                            │ Encode prompt
   │                           │                            │ Generate text
   │                           │◄───────────────────────────┤
   │                           │  JSON: {success, results}  │
   │◄──────────────────────────┤                            │
   │  Response: {              │                            │
   │    success: true,         │                            │
   │    results: [             │                            │
   │      "CC(C)C...",         │                            │
   │      "c1ccccc1..."        │                            │
   │    ]                      │                            │
   │  }                        │                            │
   │  Display results          │                            │
   └───────────────────────────┴────────────────────────────┘
```

## File Structure

```
molcrawl-web/
├── server.js                          ← Route registration
├── INFERENCE_FEATURE.md               ← Full documentation
├── INFERENCE_IMPLEMENTATION_SUMMARY.md ← Implementation summary
│
├── api/
│   └── gpt2-inference.js              ← Backend API (NEW)
│       ├── POST /api/gpt2-inference
│       ├── GET /api/gpt2-inference/config/:dataset
│       ├── getCheckpointPath()
│       ├── runInference()
│       └── DATASET_CONFIGS
│
└── src/
    ├── GPT2TrainingStatus.js          ← Main component (UPDATED)
    │   ├── handleModelClick()
    │   ├── handleModalClose()
    │   └── <InferenceModal />
    │
    ├── GPT2TrainingStatus.css         ← Styles (UPDATED)
    │   └── .model-clickable
    │
    ├── InferenceModal.js              ← Modal component (NEW)
    │   ├── Model info display
    │   ├── Parameter controls
    │   ├── handleInference()
    │   └── Results display
    │
    └── InferenceModal.css             ← Modal styles (NEW)
        ├── .inference-modal-overlay
        ├── .inference-modal-content
        ├── .inference-controls
        └── .inference-results
```

## State Management Flow

```
GPT2TrainingStatus Component State:
┌─────────────────────────────────────┐
│ modalOpen: false                    │
│ selectedModel: null                 │
└─────────────────────────────────────┘
              ↓
        User clicks card
              ↓
┌─────────────────────────────────────┐
│ modalOpen: true                     │
│ selectedModel: {                    │
│   dataset: "compounds",             │
│   size: "small",                    │
│   modelData: {...}                  │
│ }                                   │
└─────────────────────────────────────┘
              ↓
     InferenceModal renders
              ↓
┌─────────────────────────────────────┐
│ InferenceModal State:               │
│ ├── prompt: "<|start|>C"            │
│ ├── maxLength: 128                  │
│ ├── temperature: 1.0                │
│ ├── topK: null                      │
│ ├── results: []                     │
│ └── loading: false                  │
└─────────────────────────────────────┘
              ↓
      User clicks Generate
              ↓
┌─────────────────────────────────────┐
│ loading: true                       │
│ results: []                         │
└─────────────────────────────────────┘
              ↓
      API request completes
              ↓
┌─────────────────────────────────────┐
│ loading: false                      │
│ results: ["text1", "text2"]         │
└─────────────────────────────────────┘
              ↓
       Display results
```

## Error Handling Flow

```
Try: API Request
 ├─→ Success
 │   ├─→ Parse JSON
 │   ├─→ Display results
 │   └─→ Enable copy buttons
 │
 └─→ Fail
     ├─→ Network error
     │   └─→ Display: "Network request failed"
     ├─→ Model not found
     │   └─→ Display: "No checkpoint found"
     ├─→ Python error
     │   └─→ Display: stderr output
     └─→ Invalid params
         └─→ Display: "Invalid parameters"
```

## Visual States

### 1. Model Card States

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ CLICKABLE│  │ TRAINING │  │ NOT      │  │  ERROR   │
│   ✓      │  │   🚀     │  │ STARTED  │  │   ❌     │
│ (Ready)  │  │(Running) │  │ (Empty)  │  │ (Failed) │
│          │  │          │  │          │  │          │
│  Hover:  │  │  Hover:  │  │  Hover:  │  │  Hover:  │
│  Purple  │  │  Purple  │  │  None    │  │  None    │
│  Border  │  │  Border  │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 2. Modal States

```
┌─────────────────────────┐
│ Loading Configuration   │
│         ⏳              │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│ Ready to Generate       │
│   [🚀 Generate]         │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│ Generating...           │
│   [🔄 Generating...]    │
└─────────────────────────┘
           ↓
┌─────────────────────────┐
│ Results Displayed       │
│ ┌─────────────────────┐ │
│ │ Generated Text      │ │
│ │ [📋 Copy]           │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```
