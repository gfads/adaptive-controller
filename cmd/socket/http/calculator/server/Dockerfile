# Build stage
FROM golang:1.24.4 AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o server ./cmd/socket/server

# Runtime stage
FROM gcr.io/distroless/base-debian12
WORKDIR /app
COPY --from=builder /app/server .

EXPOSE 8080
CMD ["./server"]
