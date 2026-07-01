package lqr

type LQR struct {
	K float64
}

// NewPID creates a new PID controller with specified gains.
func NewLQR(k float64) *LQR {
	return &LQR{
		K: k,
	}
}

func (l *LQR) Update(cpuUsage float64) float64 {

	return -l.K * cpuUsage
}
