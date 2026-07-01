package main

import (
	"control/internal/config"
	"control/internal/controllers/def"
	"control/internal/shared"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/rabbitmq/amqp091-go"
)

type Consumer struct {
	CH         *amqp091.Channel
	Conn       amqp091.Connection
	Queue      amqp091.Queue
	CHdelivery <-chan amqp091.Delivery
	Execute    func()
}

type ErrorLine struct {
	CurrentTime      string
	Kp               string
	Ki               string
	Kd               string
	Error            string
	DerivativeError  string
	IntegrativeError string
	PC               string
	Setpoint         string
	Rate             string
	QueueSize        string
}

var Conf = &config.Config{}

func main() {

	// Load & update configuration parameters
	Conf = shared.LoadConfig(shared.Subscriber)
	if Conf.Experiment.UseEnvironmentVariables == 1 {
		shared.UpdateConfig(Conf)
	}

	// Show configuration parameters
	shared.ShowConfiguration(shared.Subscriber, *Conf)

	// Create, initialise consumer
	c := Consumer{}
	c.InitialiseRabbitMQConsumer()

	// Start consumer
	c.Start()
}

func (c *Consumer) Start() {

	forever := make(chan bool)
	nMessages := 0

	// Set initial PC
	shared.SetPC(c.CH, Conf.Experiment.PC)

	// Prepare output files
	outFile := shared.CreateOutputFile(*Conf, "data-output-"+Conf.Experiment.ControllerType+".csv")
	defer outFile.Close()

	errorsFile := shared.CreateOutputFile(*Conf, "data-errors-"+Conf.Experiment.ControllerType+".csv")
	defer errorsFile.Close()

	// Process messages in a goroutine
	mx := sync.Mutex{}
	go handleMessages(&mx, &nMessages, c)

	// Set execution type
	switch Conf.Experiment.ExecutionType {
	//ase shared.Experiment:
	//	c.Execute = c.Experiment
	case shared.OpenLoop:
		noController(&mx, &nMessages, c, outFile)
	case shared.StaticController:
		staticController(&mx, &nMessages, c, outFile)
	case shared.AdaptiveController:
		adaptiveController(&mx, &nMessages, c, outFile, errorsFile)
	default:
		shared.ErrorHandler(shared.GetFunction(), "Execution type undefined!!!")
	}
	<-forever
	return
}

func noController(mx *sync.Mutex, nMessages *int, c *Consumer, expFile *os.File) {
	d := time.Duration(Conf.Experiment.MonitorInterval)
	ticker := time.NewTicker(d * time.Second)
	nRep := 0
	for range ticker.C {
		mx.Lock()
		rate := float64(*nMessages) / float64(Conf.Experiment.MonitorInterval)
		line := strconv.Itoa(Conf.Experiment.PC) + ";" + strconv.Itoa(int(rate)) + "\n"
		fmt.Println(line)
		fmt.Fprintf(expFile, "%s", line)
		nRep++
		if nRep >= Conf.Experiment.SampleSize {
			fmt.Println("**** End of Experiment ****")
			os.Exit(0)
		}
		mx.Unlock()
	}
}

func (c *Consumer) Experiment() {
	forever := make(chan bool)
	nMessages := 0

	// Create a new controller
	ctl := def.NewController(*Conf)

	// Define initial PC
	m := make(map[int]int)
	switch Conf.Experiment.ControllerType {
	case shared.PID:
	case shared.AIMD: // AIMD + Neural
		m = shared.ReadTrainedData(Conf.NEURAL.DataTrainedPath, Conf.NEURAL.DataTrainedFile)
		Conf.Experiment.PC = shared.GetBetterPC(int(Conf.Experiment.Setpoint), m)
	case shared.Neural:
		// Read trained data - Neural
		m = shared.ReadTrainedData(Conf.NEURAL.DataTrainedPath, Conf.NEURAL.DataTrainedFile)
		Conf.Experiment.PC = shared.GetBetterPC(int(Conf.Experiment.Setpoint), m)
	default:
		shared.FailOnError(nil, "Unknown Controller Type")
	}

	// Set initial PC
	shared.SetPC(c.CH, Conf.Experiment.PC)

	// Prepare output file
	//filePath := Conf.Experiment.DockerDataDir + "/" + Conf.Experiment.ControllerType + ".csv"
	filePath := Conf.Environment.ControlDir + "/" + Conf.Experiment.ControllerType + ".csv"
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	shared.FailOnError(err, "Error opening file: %v")
	defer file.Close()

	// Process messages in a goroutine
	mx := sync.Mutex{}
	go handleMessages(&mx, &nMessages, c)

	// Invoke Controller in a goroutine
	rate := float64(0)
	sampleCounter := 0
	func() {
		d := time.Duration(Conf.Experiment.MonitorInterval)
		ticker := time.NewTicker(d * time.Second)
		for range ticker.C {
			// Calculate current rate
			rate = float64(nMessages) / float64(Conf.Experiment.MonitorInterval)

			// Show info
			qm, err := c.CH.QueueInspect(Conf.RabbitMQ.QueueName)
			if err != nil {
				log.Fatal(err)
			}

			line := strconv.Itoa(Conf.Experiment.PC) + ";" + strconv.Itoa(int(Conf.Experiment.Setpoint)) + ";" + strconv.Itoa(int(rate)) + ";" + strconv.Itoa(qm.Messages)
			fmt.Println(line) // PID

			// Save info
			_, err = fmt.Fprintln(file, line) // Write each line followed by a newline character
			if err != nil {
				log.Fatalf("Error writing to file: %v", err)
			}

			// Configure new PC
			switch Conf.Experiment.ControllerType {
			case shared.PID:
				Conf.Experiment.PC = int(math.Round(ctl.Update(Conf.Experiment.Setpoint, rate)))
			case shared.Neural: // PC is not updated
			case shared.AIMD:
				Conf.Experiment.PC = int(math.Round(ctl.Update(Conf.Experiment.Setpoint, rate)))
			case shared.HPA:
				Conf.Experiment.PC = int(math.Round(ctl.Update(Conf.Experiment.Setpoint, rate)))
			}
			shared.SetPC(c.CH, Conf.Experiment.PC)

			// Update message counting
			mx.Lock()
			nMessages = 0
			mx.Unlock()

			// Check sampling size
			sampleCounter++
			if sampleCounter >= Conf.Experiment.SampleSize {
				fmt.Println("**** Experiment Finished ****")
				os.Exit(0)
			}

			// Configure new setpoint
			idx := sampleCounter / Conf.Experiment.SameLevelSetpoint
			Conf.Experiment.Setpoint = Conf.Experiment.Setpoints[idx%len(Conf.Experiment.Setpoints)]
		}
	}()
	<-forever
}

func (c *Consumer) AdaptivePIDExperiment() {
	forever := make(chan bool)
	nMessages := 0

	// Set initial PC
	shared.SetPC(c.CH, Conf.Experiment.PC)

	// Prepare output files
	outFile := shared.CreateOutputFile(*Conf, "data-output-"+Conf.Experiment.ControllerType+".csv")
	defer outFile.Close()

	errorsFile := shared.CreateOutputFile(*Conf, "data-errors-"+Conf.Experiment.ControllerType+".csv")
	defer errorsFile.Close()

	// Process messages in a goroutine
	mx := sync.Mutex{}
	go handleMessages(&mx, &nMessages, c)

	// Invoke Controller in a goroutine
	adaptiveController(&mx, &nMessages, c, outFile, errorsFile)

	<-forever
}

func adaptiveController(mx *sync.Mutex, nMessages *int, c *Consumer, expFile *os.File, errorExpFile *os.File) {
	rate := float64(0)
	sampleCounter := 0
	integralError := 0.0
	previousError := 0.0
	currentTime := 0
	windowCount := 0

	// Create a new controller
	ctl := def.NewController(*Conf)

	d := time.Duration(Conf.Experiment.MonitorInterval)
	ticker := time.NewTicker(d * time.Second)
	errorLines := []ErrorLine{}
	errorLines = append(errorLines, ErrorLine{DerivativeError: "derivative_error", Error: "error", IntegrativeError: "integral_error"})
	for range ticker.C {
		eLine := ErrorLine{}

		// Update window count
		windowCount++
		sampleCounter++

		// Stop receiving messages using the lock
		mx.Lock()

		// Calculate current rate
		rate = float64(*nMessages) / float64(Conf.Experiment.MonitorInterval)

		// Compute errors
		currentError := Conf.Experiment.Setpoint - rate
		integralError += currentError * float64(Conf.Experiment.MonitorInterval)
		derivativeError := (currentError - previousError) / float64(Conf.Experiment.MonitorInterval)
		previousError = currentError

		// Line info
		eLine.Kp = strconv.FormatFloat(Conf.PID.Kp, 'f', 10, 64)
		eLine.Ki = strconv.FormatFloat(Conf.PID.Ki, 'f', 10, 64)
		eLine.Kd = strconv.FormatFloat(Conf.PID.Kd, 'f', 10, 64)
		eLine.Rate = strconv.Itoa(int(rate))
		eLine.Error = strconv.Itoa(int(currentError))
		eLine.IntegrativeError = strconv.Itoa(int(integralError))
		eLine.DerivativeError = strconv.Itoa(int(derivativeError))
		eLine.CurrentTime = strconv.Itoa(currentTime)
		eLine.PC = strconv.Itoa(Conf.Experiment.PC)
		eLine.Setpoint = strconv.Itoa(int(Conf.Experiment.Setpoint))

		// Build error line
		currentTime += Conf.Experiment.MonitorInterval

		// Inspect the queue size
		qm, err := c.CH.QueueInspect(Conf.RabbitMQ.QueueName)
		if err != nil {
			log.Fatal(err)
		}
		eLine.QueueSize = strconv.Itoa(qm.Messages)

		// store error info window
		errorLines = append(errorLines, eLine)

		// Save experiment output
		experimentLine := strconv.Itoa(sampleCounter) + ";" +
			eLine.PC + ";" +
			eLine.Setpoint + ";" +
			eLine.Rate + ";" +
			eLine.QueueSize
		_, err = fmt.Fprintln(expFile, experimentLine) // Write each line followed by a newline character
		if err != nil {
			log.Fatalf("Error writing to file: %v", err)
		}

		// Show info
		fmt.Println(experimentLine) // PID

		// update gains
		if windowCount >= Conf.Experiment.AdaptiveWindowSize {
			//previousError = 0.0
			//integralError = 0.0
			windowCount = 0
			currentTime = 0

			// Prepare error file
			errorsFilePath := Conf.Python.ErrorsFile
			//errorsFile, err := os.OpenFile(errorsFilePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
			errorsFile, err := os.Create(errorsFilePath)
			if err != nil {
				shared.ErrorHandler(shared.GetFunction(), "Error opening file:"+Conf.Python.ErrorsFile)
			}
			for _, line := range errorLines {

				// Update *partial* error file
				_, err := fmt.Fprintln(errorsFile,
					line.Error+","+
						line.DerivativeError+","+
						line.IntegrativeError)
				if err != nil {
					log.Fatalf("Error writing to file: %v", err)
				}

				// Update *complete* error file
				if !strings.Contains(line.Error, "error") { // filter header lines
					_, err = fmt.Fprintln(errorExpFile,
						line.Error+","+
							line.DerivativeError+","+
							line.IntegrativeError,
					)
					if err != nil {
						log.Fatalf("Error writing to file: %v", err)
					}
				}
			}

			// Remove previous lines & add header & close file
			errorLines = []ErrorLine{} // remove previous errors
			errorLines = append(errorLines, ErrorLine{DerivativeError: "derivative_error", Error: "error", IntegrativeError: "integral_error"})
			errorsFile.Close()

			fmt.Printf("**** Old Gains: kp=%.10f, ki=%.10f, kd=%.10f *** \n",
				Conf.PID.Kp, Conf.PID.Ki, Conf.PID.Kd)

			// Adjustment mechanism (Python)
			Conf.PID.Kp, Conf.PID.Ki, Conf.PID.Kd = shared.GetNNGains(Conf.Python.Path,
				Conf.Python.Executable,
				Conf.Python.Script,
				Conf.Python.ErrorsFile,
				Conf.Python.GainsFile,
				Conf.PID.Kp,
				Conf.PID.Ki,
				Conf.PID.Kd,
				Conf.Python.Mode,
				Conf.Python.JitterOn)

			// Configure new gains
			ctl.Init(Conf.PID.Direction,
				Conf.PID.Kp,
				Conf.PID.Ki,
				Conf.PID.Kd,
				Conf.PID.Min,
				Conf.PID.Max)

			fmt.Printf("**** New Gains: kp=%.10f, ki=%.10f, kd=%.10f *** \n",
				Conf.PID.Kp, Conf.PID.Ki, Conf.PID.Kd)
			ticker.Reset(d * time.Second) // reset time
		}

		// Update pc
		Conf.Experiment.PC = int(math.Round(ctl.Update(Conf.Experiment.Setpoint, rate)))
		shared.SetPC(c.CH, Conf.Experiment.PC)

		// Update message counting & unlock
		*nMessages = 0
		mx.Unlock()

		// Check sampling size
		//sampleCounter++
		if sampleCounter >= Conf.Experiment.SampleSize {
			fmt.Println("**** Experiment Finished ****")
			os.Exit(0)
		}

		// Configure new setpoint
		idx := sampleCounter / Conf.Experiment.SameLevelSetpoint
		Conf.Experiment.Setpoint = Conf.Experiment.Setpoints[idx%len(Conf.Experiment.Setpoints)]
	}
}

func staticController(mx *sync.Mutex, nMessages *int, c *Consumer, expFile *os.File) {
	rate := float64(0)
	sampleCounter := 0

	// Create a new controller
	ctl := def.NewController(*Conf)

	d := time.Duration(Conf.Experiment.MonitorInterval)
	ticker := time.NewTicker(d * time.Second)
	for range ticker.C {
		// Stop receiving messages using the lock
		mx.Lock()

		// Calculate current rate
		rate = float64(*nMessages) / float64(Conf.Experiment.MonitorInterval)

		// Inspect the queue size
		qm, err := c.CH.QueueInspect(Conf.RabbitMQ.QueueName)
		if err != nil {
			log.Fatal(err)
		}

		// Save experiment output
		experimentLine := strconv.Itoa(sampleCounter+1) + ";" +
			strconv.Itoa(Conf.Experiment.PC) + ";" +
			strconv.Itoa(int(Conf.Experiment.Setpoint)) + ";" +
			strconv.Itoa(int(rate)) + ";" +
			strconv.Itoa(qm.Messages)
		_, err = fmt.Fprintln(expFile, experimentLine) // Write each line followed by a newline character
		if err != nil {
			log.Fatalf("Error writing to file: %v", err)
		}

		// Show info
		fmt.Println(experimentLine) // PID

		// Update pc
		Conf.Experiment.PC = int(math.Round(ctl.Update(Conf.Experiment.Setpoint, rate)))
		shared.SetPC(c.CH, Conf.Experiment.PC)

		// Update message counting
		*nMessages = 0
		mx.Unlock()

		// Check sampling size
		sampleCounter++
		if sampleCounter >= Conf.Experiment.SampleSize {
			fmt.Println("**** Experiment Finished ****")
			os.Exit(0)
		}

		// Configure new setpoint
		idx := sampleCounter / Conf.Experiment.SameLevelSetpoint
		Conf.Experiment.Setpoint = Conf.Experiment.Setpoints[idx%len(Conf.Experiment.Setpoints)]
	}
}

func (c *Consumer) OpenLoop() {
	forever := make(chan bool)
	nMessages := 0

	// Show configuration parameters
	shared.ShowConfiguration(shared.Subscriber, *Conf)

	// Set initial prefetch count
	shared.SetPC(c.CH, Conf.Experiment.PC)

	// Process messages in a goroutine
	mx := sync.Mutex{}
	go handleMessages(&mx, &nMessages, c)

	func() {
		mx := sync.Mutex{}
		d := time.Duration(Conf.Experiment.MonitorInterval)
		ticker := time.NewTicker(d * time.Second)
		nRep := 0
		sumRate := 0.0
		for range ticker.C {
			//fmt.Println(pc, ";", nMessages/10, ";", getQueueLength(shared.QueueName))
			sumRate += float64(nMessages) / float64(Conf.Experiment.MonitorInterval)
			//fmt.Println(Conf.Experiment.PC, ";", nMessages/int(Conf.Experiment.MonitorInterval))
			mx.Lock()
			nMessages = 0
			mx.Unlock()
			nRep++
			if nRep > Conf.Experiment.SameLevelPc { // TODO
				fmt.Println(Conf.Experiment.PC, ";", int(sumRate/float64(Conf.Experiment.SameLevelPc)))
				nRep = 0
				sumRate = 0
				Conf.Experiment.PC += Conf.Experiment.PCStep // step pc increment

				if Conf.Experiment.PC > Conf.Experiment.MaxTrainingPC {
					fmt.Println("**** End of Experiment ****")
					os.Exit(0)
				}
				// update pc
				shared.SetPC(c.CH, Conf.Experiment.PC)
			}
		}
	}()

	<-forever
}

func (c *Consumer) TrainingExperiment() {
	forever := make(chan bool)
	nMessages := 0

	// Prepare output file
	filePath := Conf.Experiment.DockerDataDir + "/" + Conf.NEURAL.DataTrainingFile
	//filePath := Conf.Environment.ControlDir + "/" + Conf.NEURAL.DataTrainingFile
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	shared.FailOnError(err, "Error opening file:"+filePath)
	defer file.Close() // Ensure the file is closed when the function exits

	// Generate PC using a Sin cruve
	pcs := shared.GenerateSinPC()

	// Set initial prefetch count
	shared.SetPC(c.CH, pcs[0])

	// Show configuration parameters
	shared.ShowConfiguration(shared.Subscriber, *Conf)

	log.Println(" [*] Waiting for messages. To exit press CTRL+C [" + shared.GetFunction() + "]")

	// Process messages in a goroutine
	mx := sync.Mutex{}
	go handleMessages(&mx, &nMessages, c)

	// Update PC
	Conf.Experiment.PC = pcs[0]
	func() {
		mx := sync.Mutex{}
		d := time.Duration(Conf.Experiment.MonitorInterval)
		ticker := time.NewTicker(d * time.Second)
		idx := 0
		for range ticker.C {
			// calculate rate
			rate := float64(nMessages) / float64(Conf.Experiment.MonitorInterval)

			// Show info
			line := strconv.Itoa(Conf.Experiment.PC) + ";" + strconv.Itoa(int(rate))
			fmt.Println(line) // PID

			// Save info
			_, err := fmt.Fprintln(file, line)
			if err != nil {
				log.Fatalf("Error writing to file: %v", err)
			}

			mx.Lock()
			nMessages = 0
			mx.Unlock()

			// new pc
			Conf.Experiment.PC = pcs[idx] // randon pc

			// Configure pc
			shared.SetPC(c.CH, Conf.Experiment.PC)

			idx++
			if idx >= len(pcs) {
				fmt.Println("********** End of experiment ***********")
				os.Exit(0)
			}
		}
	}()
	<-forever
}

func handleMessages(mx *sync.Mutex, nMessages *int, c *Consumer) {

	log.Println(" [*] Waiting for messages. To exit press CTRL+C [" + shared.GetFunction() + "]")
	for d := range c.CHdelivery {
		err := d.Ack(false)
		shared.FailOnError(err, "Failed to Ack a message")
		mx.Lock()
		*nMessages++
		mx.Unlock()
	}
}

func (c *Consumer) InitialiseRabbitMQConsumer() {
	// Connect to RabbitMQ
	cx, err := amqp091.Dial("amqp://" + Conf.RabbitMQ.User + ":" + Conf.RabbitMQ.Pwd + "@" + Conf.RabbitMQ.Host + ":" + strconv.Itoa(Conf.RabbitMQ.Port) + "/")
	shared.FailOnError(err, "Failed to connect to RabbitMQ")
	c.Conn = *cx

	// Initialise the channel
	c.CH, err = c.Conn.Channel()
	shared.FailOnError(err, "Failed to open a channel")

	// Set max queue size
	args := amqp091.Table{
		"x-max-length": int32(Conf.Experiment.QueueMaxLenght), // Set the maximum length to 100K messages
	}

	// Declare the queue (must match publisher)
	c.Queue, err = c.CH.QueueDeclare(
		Conf.RabbitMQ.QueueName, // name
		true,                    // durable
		true,                    // delete when unused
		false,                   // exclusive
		false,                   // no-wait
		args,                    // arguments
	)
	shared.FailOnError(err, "Failed to declare a queue")

	// Define role as a consumer
	cd, err := c.CH.Consume(
		Conf.RabbitMQ.QueueName, // queue
		"",                      // consumer tag
		false,                   // auto-ack
		false,                   // exclusive
		false,                   // no-local
		false,                   // no-wait
		nil,                     // args
	)
	shared.FailOnError(err, "Failed to register a consumer")
	c.CHdelivery = cd
}
