package onoff

type OnOff struct {
	Max float64
	Min float64
}

func Init(min, max float64) OnOff {
	c := OnOff{Max: max, Min: min}
	return c
}

func (c OnOff) Update(g float64, m float64) float64 {
	if m > g {
		return c.Min
	} else {
		return c.Max
	}
}
