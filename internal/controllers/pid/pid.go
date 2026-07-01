package pid

import (
	"time"
)

type PID struct {
	Kp, Ki, Kd float64 // Tuning parameters
	Max        float64
	Min        float64
	Direction  float64
	prevError  float64
	integral   float64
	lastTime   time.Time
}

// NewPID creates a new PID controller with specified gains.
func (c *PID) Init(p ...float64) {
	c.Direction = p[0]
	c.Kp = p[1]
	c.Ki = p[2]
	c.Kd = p[3]
	c.Min = p[4]
	c.Max = p[5]
	c.prevError = 0.0
	c.integral = 0.0
	c.lastTime = time.Now()
}

// Update calculates the control-theory variable from the error and returns it.
func (c *PID) Update(p ...float64) float64 {
	setpoint := p[0]
	measured := p[1]
	now := time.Now()
	dt := now.Sub(c.lastTime).Seconds()

	// prevent division by zero
	if dt <= 0 {
		dt = 1e-3
	}

	e := c.Direction * (setpoint - measured)

	c.integral += e * dt
	derivative := (e - c.prevError) / dt

	output := c.Kp*e + c.Ki*c.integral + c.Kd*derivative

	// apply max/min constraints
	if output > c.Max {
		output = c.Max
	} else if output < c.Min {
		output = c.Min
	}

	// Save for next update
	c.prevError = e
	c.lastTime = now

	return output
}
