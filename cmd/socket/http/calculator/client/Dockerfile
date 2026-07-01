# Build stage
FROM golang:1.24.4 AS builder
WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY ../../.. .
#RUN ls -R /app
#CMD ["sh"]

RUN CGO_ENABLED=0 GOOS=linux go build -o client ./cmd/socket/client
#RUN go build -o client ./cmd/socket-client

# Runtime stage
FROM gcr.io/distroless/base-debian12
WORKDIR /app
COPY --from=builder /app/client .

CMD ["./client"]
