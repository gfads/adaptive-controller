package hpa

import (
	"math"
)

type HPA struct {
	Direction float64
	Min       float64
	Max       float64
	PC        float64
}

func (c *HPA) Init(p ...float64) {
	c.Direction = p[0]
	c.Min = p[1]
	c.Max = p[2]
	c.PC = p[3]
}

func (c *HPA) Update(p ...float64) float64 {
	u := 0.0

	s := p[0] // goal
	y := p[1] // plant output

	// control law
	u = math.Round(c.PC * s / y)

	if u > c.Max {
		u = c.Max
	} else if u < c.Min {
		u = c.Min
	}

	c.PC = u

	return u
}
