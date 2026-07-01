package main

import (
	"control/internal/config"
	"control/internal/shared"
	"fmt"
	"math/rand"
	"strconv"
	"sync"
	"time"

	"github.com/rabbitmq/amqp091-go"
)

type Publisher struct{}

var Conf = &config.Config{}

func main() {

	// Load configuration parameters
	shared.CheckConfFile()
	err := *new(error)
	Conf = shared.LoadConfig(shared.Publisher)
	if err != nil {
		shared.ErrorHandler(shared.GetFunction(), "Error on loading "+shared.ConfigFile)
	}

	fmt.Println("****** Here ******", Conf.Publisher.ExecutionEnv)

	// Update config with env variables (for EC2 environment)
	if Conf.Experiment.UseEnvironmentVariables == 1 { //1=Yes, //0=No
		shared.UpdateConfig(Conf)
	}

	// Show basic information
	li := (time.Duration(Conf.Experiment.PublishInterval)*time.Millisecond + time.Duration(Conf.Experiment.PublishIntervalSTD)*time.Millisecond).Seconds()
	ls := (time.Duration(Conf.Experiment.PublishInterval)*time.Millisecond - time.Duration(Conf.Experiment.PublishIntervalSTD)*time.Millisecond).Seconds()
	fmt.Println("******************* PUBLISHER **********************")
	fmt.Println("Execution Environment : ", Conf.Publisher.ExecutionEnv)
	fmt.Println("RabbitMQ Host         : ", Conf.RabbitMQ.Host)
	fmt.Println("Publishers Type       : ", Conf.Experiment.PublishersType)
	fmt.Println("Number of Publishers  : ", Conf.Experiment.NumberPublishers)
	fmt.Println("Publish Interval      : ", Conf.Experiment.PublishInterval, "ms")
	fmt.Println("Publ ish Interval (STD): ", Conf.Experiment.PublishIntervalSTD, "ms")
	fmt.Printf("Publication Rate[~]   : [%.f,%.f] msgs/s\n", float64(Conf.Experiment.NumberPublishers)/li, float64(Conf.Experiment.NumberPublishers)/ls) // TODO
	fmt.Println("*****************************************")

	// New publisher
	p := Publisher{}
	p.Run()
}

func (p *Publisher) Run() {
	// Connect to RabbitMQ
	conn, err := amqp091.Dial("amqp://" + Conf.RabbitMQ.User + ":" + Conf.RabbitMQ.Pwd + "@" + Conf.RabbitMQ.Host + ":" + strconv.Itoa(Conf.RabbitMQ.Port) + "/")
	shared.FailOnError(err, "Failed to connect to RabbitMQ")
	defer conn.Close()

	// Open a channel
	ch, err := conn.Channel()
	shared.FailOnError(err, "Failed to open a channel")
	defer ch.Close()

	// max queue size
	args := amqp091.Table{
		"x-max-length": int32(Conf.Experiment.QueueMaxLenght), // Set the maximum length to 10 messages
	}

	// Declare a queue (must match subscriber)
	_, err = ch.QueueDeclare(
		Conf.RabbitMQ.QueueName, // name
		true,                    // durable
		true,                    // delete when unused
		false,                   // exclusive
		false,                   // no-wait
		args,                    // arguments
	)
	shared.FailOnError(err, "Failed to declare a queue")

	switch Conf.Experiment.PublishersType {
	case shared.Randon:
		randonPublishers(ch)
	case shared.Fixed:
		fixedPublishers(ch)
	default:
		shared.ErrorHandler(shared.GetFunction(), "Publishers Type Unknown: "+Conf.Experiment.PublishersType)
	}
}

// Manager controls a dynamic set of worker goroutines.
type Manager struct {
	mu      sync.Mutex
	nextID  int
	workers map[int]chan struct{} // id -> stop channel
}

func NewManager() *Manager {
	return &Manager{
		workers: make(map[int]chan struct{}),
	}
}

func (m *Manager) AddPublisher(ch *amqp091.Channel) {
	m.mu.Lock()
	defer m.mu.Unlock()

	id := m.nextID
	m.nextID++

	stopCh := make(chan struct{})
	m.workers[id] = stopCh

	go publisher(id, stopCh, ch)
}

func (m *Manager) RemovePublisher() {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.workers) == Conf.Experiment.MinNumberOfPublishers {
		return
	}

	for id, ch := range m.workers {
		close(ch)
		delete(m.workers, id)
		break
	}
}

func publisher(id int, stopCh <-chan struct{}, ch *amqp091.Channel) {
	for {
		select {
		case <-stopCh:
			return
		default:
			body := "XXXXXXXX" + time.Now().Format(time.RFC3339)
			err := ch.Publish(
				"",                      // exchange
				Conf.RabbitMQ.QueueName, // routing key (queue name)
				false,                   // mandatory
				false,                   // immediate
				amqp091.Publishing{
					ContentType: "text/plain",
					Body:        []byte(body),
				})
			shared.FailOnError(err, "Failed to publish a message")
			time.Sleep(shared.NormallyDistributedDuration(time.Duration(Conf.Experiment.PublishInterval)*time.Millisecond, time.Duration(Conf.Experiment.PublishIntervalSTD)*time.Millisecond))
		}
	}
}

func fixedPublishers(ch *amqp091.Channel) {
	// Start publishing
	forever := make(chan bool)
	fmt.Println("Publisher: Start publishing.. ")
	for i := 0; i < Conf.Experiment.NumberPublishers; i++ {
		go func() {
			for {
				// Create & send the message
				body := "XXXXXXXX" + time.Now().Format(time.RFC3339)
				err := ch.Publish(
					"",                      // exchange
					Conf.RabbitMQ.QueueName, // routing key (queue name)
					false,                   // mandatory
					false,                   // immediate
					amqp091.Publishing{
						ContentType: "text/plain",
						Body:        []byte(body),
					})
				shared.FailOnError(err, "Failed to publish a message")

				time.Sleep(shared.NormallyDistributedDuration(time.Duration(Conf.Experiment.PublishInterval)*time.Millisecond, time.Duration(Conf.Experiment.PublishIntervalSTD)*time.Millisecond))
				//log.Printf(" [x] Sent %s", body)
			}
		}()
	}
	<-forever
}

func randonPublishers(ch *amqp091.Channel) {
	rand.Seed(time.Now().UnixNano())

	// Default initial number of workers
	initialPublishers := Conf.Experiment.NumberPublishers

	mgr := NewManager()

	// Start initial pool of workers
	for i := 0; i < initialPublishers; i++ {
		mgr.AddPublisher(ch)
	}

	// Periodically change the number of workers forever
	ticker := time.NewTicker(time.Duration(Conf.Experiment.IncDecInterval) * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		// Random delta in {-1, 0, +1}
		decision := rand.Intn(3) - 1
		delta := rand.Intn(Conf.Experiment.IncDecPublishersStep) // up to

		switch {
		case decision > 0:
			for i := 0; i < delta; i++ {
				mgr.AddPublisher(ch)
			}
		case decision < 0:
			for i := 0; i < delta; i++ {
				mgr.RemovePublisher()
			}
		default:
			// No change this tick
		}
	}
}
