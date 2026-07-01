# Build stage
FROM golang:1.24.4 AS builder
WORKDIR /app

# Copy go.mod and go.sum first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the whole project (so shared packages are included)
COPY . .

# Build pub app
RUN go build -o pub ./cmd/rabbitmq/pub

# Runtime stage
FROM gcr.io/distroless/base-debian12

COPY --from=builder /app/pub /app/pub
COPY ./internal/config/config.json /app

WORKDIR /app
CMD ["./pub"]
