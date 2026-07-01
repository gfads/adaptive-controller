package def

import (
	"control/internal/config"
	"control/internal/controllers/aimd"
	"control/internal/controllers/hpa"
	"control/internal/controllers/pid"
	"control/internal/shared"
)

type Controller interface {
	Init(p ...float64)
	Update(p ...float64) float64
}

func NewController(conf config.Config) Controller {
	switch conf.Experiment.ControllerType {
	case shared.PID:
		c := pid.PID{}
		c.Init(conf.PID.Direction,
			conf.PID.Kp,
			conf.PID.Ki,
			conf.PID.Kd,
			conf.PID.Min,
			conf.PID.Max)
		return &c
	case shared.AdaptivePID:
		c := pid.PID{}
		c.Init(conf.PID.Direction,
			conf.PID.Kp,
			conf.PID.Ki,
			conf.PID.Kd,
			conf.PID.Min,
			conf.PID.Max)
		return &c
	case shared.AIMD:
		c := aimd.AIMD{}
		c.Init(conf.AIMD.Min,
			conf.AIMD.Max,
			conf.AIMD.HysterisBand,
			conf.AIMD.PreviousOut,
			conf.AIMD.PreviousRate)
		return &c
	case shared.HPA:
		c := hpa.HPA{}
		c.Init(conf.HPA.Direction,
			conf.HPA.Min,
			conf.HPA.Max,
			conf.HPA.PC)
		return &c
	default:
		shared.FailOnError(nil, "Unsupported controller type")
	}

	return *new(Controller)
}
