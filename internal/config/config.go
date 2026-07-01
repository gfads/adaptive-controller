package config

type Config struct {
	Environment EnvironmentConf `json:"environment"`
	RabbitMQ    RabbitmqConf    `json:"rabbitmq"`
	Publisher   PublisherConf   `json:"publisher"`
	Subscriber  SubscriberConf  `json:"subscriber"`
	PID         PIDConf         `json:"pid"`
	NEURAL      NeuralConf      `json:"neural"`
	AIMD        AIMDConf        `json:"aimd"`
	HPA         HPAConf         `json:"hpa"`
	Experiment  ExpConf         `json:"experiment"`
	Python      PythonConf      `json:"python"`
}

type EnvironmentConf struct {
	ControlDir string `json:"controldir"`
	GoRoot     string `json:"goroot"`
}
type RabbitmqConf struct {
	Host         string `json:"host"`
	Port         int    `json:"port"`
	User         string `json:"user"`
	Pwd          string `json:"pwd"`
	QueueName    string `json:"queue-name"`
	ExecutionEnv string `json:"execution-env"`
}

type PythonConf struct {
	Path       string `json:"path"`
	ErrorsFile string `json:"errorsFile"`
	GainsFile  string `json:"gainsFile"`
	Executable string `json:"executable"`
	Script     string `json:"script"`
	Mode       string `json:"mode"`
	Dir        string `json:"dir"`
	JitterOn   string `json:"jitterOn"`
}

type PublisherConf struct {
	Host         string `json:"host"`
	ExecutionEnv string `json:"execution-env"`
}

type SubscriberConf struct {
	Host         string `json:"host"`
	ExecutionEnv string `json:"execution-env"`
}

type PIDConf struct {
	Direction float64 `json:"direction"`
	Kp        float64 `json:"kp"`
	Ki        float64 `json:"ki"`
	Kd        float64 `json:"kd"`
	Max       float64 `json:"max"`
	Min       float64 `json:"min"`
}

type AIMDConf struct {
	Min          float64 `json:"min"`
	Max          float64 `json:"max"`
	HysterisBand float64 `json:"hysteris-band"`
	PreviousRate float64 `json:"previous-rate"`
	PreviousOut  float64 `json:"previous-out"`
}

type HPAConf struct {
	Direction float64 `json:"direction"`
	Min       float64 `json:"min"`
	Max       float64 `json:"max"`
	PC        float64 `json:"pc"`
}

type NeuralConf struct {
	DataTrainedPath  string `json:"data-trained-path"`
	DataTrainedFile  string `json:"data-trained-file"`
	DataTrainingPath string `json:"data-training-path"`
	DataTrainingFile string `json:"data-training-file"`
}

type ExpConf struct {
	ExecutionType           string    `json:"execution-type"`
	SameLevelPc             int       `json:"same-level-pc"`
	SameLevelSetpoint       int       `json:"same-level-setpoint"`
	Setpoint                float64   `json:"setpoint"`
	PC                      int       `json:"pc"`
	MonitorInterval         int       `json:"monitor-interval"`
	ControllerType          string    `json:"controller-type"`
	DockerDataDir           string    `json:"docker-data-dir"`
	DockerDataFile          string    `json:"docker-data-file"`
	SampleSize              int       `json:"sample-size"`
	PublishInterval         int       `json:"publish-interval"`
	PublishIntervalSTD      int       `json:"publish-interval-std"`
	PCStep                  int       `json:"pc-step"`
	MaxTrainingPC           int       `json:"max-training-pc"`
	AdaptiveWindowSize      int       `json:"adaptive-window-size"`
	Setpoints               []float64 `json:"setpoints"`
	QueueMaxLenght          int       `json:"queue-max-lenght"`
	NumberPublishers        int       `json:"number-publishers"`
	IncDecPublishersStep    int       `json:"inc-dec-publishers-step"`
	IncDecInterval          int       `json:"inc-dec-interval"`
	MinNumberOfPublishers   int       `json:"min-number-publishers"`
	PublishersType          string    `json:"publishers-type"`
	UseEnvironmentVariables int       `json:"use-environment-variables"` // 1= Yes, 0=No
}
