# 🧩 Building a Local LLM Inference API for Semantic Similarity

At **[whatisgoing.com](https://whatisgoing.com)**, we needed a reliable way to **extract named entities** from multilingual news articles — especially Arabic news — where the same entity can appear in different linguistic forms.

For example:

> "انتصارات أكتوبر" and "حرب أكتوبر"

…both refer to the same historical event — the 1973 war between Egypt and Israel — yet share no overlapping words.  
Traditional keyword or similarity-based methods fail to capture such **semantic equivalence**.

---

## 🎯 The Challenge

We wanted a solution that could:

- Understand **multilingual context** (Arabic, English, German)
- Calculate **semantic similarity** between text pairs
- Decide if two texts **refer to the same entity or event**
- Return a **structured JSON** result for use in our core analytics engine
- Work **locally**, without depending on external APIs or slow cloud inference

---

## 🌍 Examples of the Challenge

<summary>🇸🇦 Arabic Examples</summary>

| Example | Explanation |
|---------|-------------|
| `انتصارات أكتوبر` vs `حرب أكتوبر` | Both refer to the October War (1973), but use different words. |
| `عبد الفتاح السيسي` vs `الرئيس السيسي` | One uses the full name; the other uses a title with surname. |


<summary>🇬🇧 English Examples</summary>

| Example | Explanation |
|---------|-------------|
| `President Abdel Fattah el-Sisi` vs `El Sisi` vs `President Sisi` | All refer to the same person, phrased differently. |
| `United Nations` vs `UN` vs `the UN organization` | Various forms of the same entity. |

<summary>🇩🇪 German Examples</summary>

| Example | Explanation |
|---------|-------------|
| `Bundeskanzler Olaf Scholz` vs `Scholz` vs `der Kanzler` | Different references (title, surname, or role) to the same person. |
| `Europäische Union` vs `EU` | Long and short forms of the same organization. |
---


## 🏗️ Architecture Overview

Our service follows a clean, layered architecture that separates concerns and ensures maintainability:

```
llm-inference/
├── main.go                 # Application bootstrap
├── config/                 # Configuration management
├── models/                 # Data structures and types
├── services/               # Business logic and LLM calls
├── handlers/               # HTTP request handlers
└── routes/                 # Route definitions and middleware
```

### Core Components

1. **Configuration Layer**: Environment-driven settings
2. **Models Layer**: Strongly-typed request/response structures
3. **Services Layer**: Business logic and Ollama API integration
4. **Handlers Layer**: HTTP request processing
5. **Routes Layer**: Endpoint definitions and middleware

---

## 🔧 Implementation Deep Dive

### 1. Configuration Management

First, we define our configuration structure to handle environment variables and defaults:

```go
// config/config.go
package config

import (
    "os"
    "github.com/joho/godotenv"
)

type Config struct {
    OllamaURL string
    ModelName string
    APIUrl    string
    Port      string
    GinMode   string
}

func Load() *Config {
    godotenv.Load()
    
    cfg := &Config{
        OllamaURL: getEnv("OLLAMA_URL", "http://localhost:11434"),
        ModelName: getEnv("MODEL_NAME", "deepseek-r1:1.5b"),
        Port:      getEnv("PORT", "8090"),
        GinMode:   getEnv("GIN_MODE", "release"),
    }
    cfg.APIUrl = cfg.OllamaURL + "/api/generate"
    return cfg
}

func getEnv(key, defaultValue string) string {
    if value := os.Getenv(key); value != "" {
        return value
    }
    return defaultValue
}
```

### 2. Data Models

We define strongly-typed structures for all API interactions:

```go
// models/models.go
package models

type InputText struct {
    Text string `json:"text" binding:"required"`
}

type SimilarityResponse struct {
    Text1           string  `json:"text1"`
    Text2           string  `json:"text2"`
    SimilarityScore float64 `json:"similarity_score"`
    ShouldMerge     bool    `json:"should_be_merged"`
    Thinking        string  `json:"thinking,omitempty"`
}

type OllamaRequest struct {
    Model  string `json:"model"`
    Prompt string `json:"prompt"`
    Stream bool   `json:"stream"`
}

type OllamaResponse struct {
    Response     string `json:"response"`
    Thinking     string `json:"thinking"`
    JSONResponse string `json:"json_response"`
}
```

### 3. Service Layer

The service layer handles all business logic and external API calls:

```go
// services/services.go
package services

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "strings"
    "llm-inference/config"
    "llm-inference/models"
)

type OllamaService struct {
    config *config.Config
}

func NewOllamaService(cfg *config.Config) *OllamaService {
    return &OllamaService{config: cfg}
}

func (s *OllamaService) CallAPI(prompt string) (string, error) {
    reqBody := models.OllamaRequest{
        Model:  s.config.ModelName,
        Prompt: prompt,
        Stream: false,
    }

    jsonData, err := json.Marshal(reqBody)
    if err != nil {
        return "", fmt.Errorf("failed to marshal request: %v", err)
    }

    resp, err := http.Post(s.config.APIUrl, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return "", fmt.Errorf("failed to call Ollama API: %v", err)
    }
    defer resp.Body.Close()

    var ollamaResp models.OllamaResponse
    if err := json.NewDecoder(resp.Body).Decode(&ollamaResp); err != nil {
        return "", fmt.Errorf("failed to decode response: %v", err)
    }

    return strings.TrimSpace(ollamaResp.Response), nil
}

type SimilarityService struct {
    ollama *OllamaService
}

func NewSimilarityService(ollama *OllamaService) *SimilarityService {
    return &SimilarityService{ollama: ollama}
}

func (s *SimilarityService) AnalyzeSimilarity(text1, text2 string) (*models.SimilarityResponse, error) {
    prompt := fmt.Sprintf(`
Analyze the semantic similarity between these two texts and determine if they should be merged.

Text 1: "%s"
Text 2: "%s"

Provide your reasoning and return a JSON response with this structure:
{
  "text1": "first text",
  "text2": "second text", 
  "similarity_score": 0.85,
  "should_be_merged": true
}

Score should be between 0.0 (completely different) and 1.0 (identical).
Consider them mergeable if similarity > 0.7.`, text1, text2)

    response, err := s.ollama.CallAPI(prompt)
    if err != nil {
        return nil, err
    }

    // Parse the LLM response and extract JSON
    var result models.SimilarityResponse
    if err := json.Unmarshal([]byte(response), &result); err != nil {
        // Handle cases where LLM includes reasoning text
        return s.extractJSONFromResponse(response, text1, text2)
    }

    return &result, nil
}
```

### 4. HTTP Handlers

Handlers manage HTTP requests, validation, and responses:

```go
// handlers/handlers.go
package handlers

import (
    "fmt"
    "net/http"
    "time"
    "llm-inference/models"
    "llm-inference/services"
    "github.com/gin-gonic/gin"
)

type Handler struct {
    similarityService *services.SimilarityService
}

func NewHandler(similarityService *services.SimilarityService) *Handler {
    return &Handler{similarityService: similarityService}
}

func (h *Handler) AnalyzeSimilarity(c *gin.Context) {
    var input struct {
        Text1 string `json:"text1" binding:"required"`
        Text2 string `json:"text2" binding:"required"`
    }

    if err := c.ShouldBindJSON(&input); err != nil {
        c.JSON(http.StatusBadRequest, models.ErrorResponse{Error: err.Error()})
        return
    }

    fmt.Printf("Analyzing similarity between texts (%.50s... vs %.50s...)\n", 
               input.Text1, input.Text2)

    startTime := time.Now()
    result, err := h.similarityService.AnalyzeSimilarity(input.Text1, input.Text2)
    if err != nil {
        c.JSON(http.StatusInternalServerError, 
               models.ErrorResponse{Error: "Failed to analyze similarity"})
        return
    }

    elapsed := time.Since(startTime)
    fmt.Printf("Similarity analysis completed in %.2f seconds\n", elapsed.Seconds())

    c.JSON(http.StatusOK, result)
}
```

### 5. Application Bootstrap

The main function ties everything together with dependency injection:

```go
// main.go
package main

import (
    "fmt"
    "log"
    "llm-inference/config"
    "llm-inference/handlers"
    "llm-inference/routes"
    "llm-inference/services"
    "github.com/gin-gonic/gin"
)

func main() {
    cfg := config.Load()
    gin.SetMode(cfg.GinMode)

    ollamaService := services.NewOllamaService(cfg)
    similarityService := services.NewSimilarityService(ollamaService)
    handler := handlers.NewHandler(similarityService)

    router := routes.SetupRouter(handler)

    fmt.Println("Starting LLM Inference Service v1.2.0")
    fmt.Printf("Ollama URL: %s\n", cfg.OllamaURL)
    fmt.Printf("Model: %s\n", cfg.ModelName)
    fmt.Printf("Server starting on :%s\n", cfg.Port)

    if err := router.Run(":" + cfg.Port); err != nil {
        log.Fatal("Failed to start server:", err)
    }
}
```

---

## 🌐 API Usage Examples

### Text Similarity Analysis

**Request:**
```bash
curl -X POST http://localhost:8090/api/v1/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "The quick brown fox jumps over the lazy dog",
    "text2": "A fast brown fox leaps over a sleepy dog"
  }'
```

**Response:**
```json
{
  "text1": "The quick brown fox jumps over the lazy dog",
  "text2": "A fast brown fox leaps over a sleepy dog",
  "similarity_score": 0.87,
  "should_be_merged": true,
  "thinking": "Both texts describe the same scenario with similar vocabulary and structure."
}
```

### Health Check

**Request:**
```bash
curl http://localhost:8090/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "LLM Inference Service",
  "uptime": "2025-10-05T14:30:00Z",
  "version": "1.2.0"
}
```

### Service Information

**Request:**
```bash
curl http://localhost:8090/
```

**Response:**
```json
{
  "title": "LLM Inference Service",
  "description": "High-performance API for LLM-powered text analysis",
  "version": "1.2.0",
  "endpoints": {
    "similarity": "/api/v1/similarity",
    "health": "/health"
  }
}
```

---

## 🚀 Deployment & Running

### Local Development

1. **Install Ollama** and pull the required model:
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull deepseek-r1:1.5b
```

2. **Set up environment variables** (`.env` file):
```env
OLLAMA_URL=http://localhost:11434
MODEL_NAME=deepseek-r1:1.5b
PORT=8090
GIN_MODE=debug
```

3. **Run the service**:
```bash
go run main.go
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o llm-inference .

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/llm-inference .
EXPOSE 8090
CMD ["./llm-inference"]
```

**Build and run:**
```bash
docker build -t llm-inference .
docker run -p 8090:8090 --env-file .env llm-inference
```

---

## 📝 Conclusion

Building an LLM inference API service in Go provides significant advantages in performance, maintainability, and deployment simplicity. The modular architecture ensures your service can scale from prototype to production while maintaining clean, testable code.

The complete source code and deployment examples are available in the [GitHub repository](https://github.com/whatisgoing-com/llm-inference).

---

## 🏷️ Tags
`#golang` `#llm` `#ai` `#api` `#ollama` `#microservices` `#nlp` `#machinelearning` `#backend` `#softwaredevelopment`