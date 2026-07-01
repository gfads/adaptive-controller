package mpc

import (
	"control/internal/shared"
	"os/exec"
	"strconv"
	"strings"
)

type MPC struct {
	A        float64
	B        float64
	Setpoint float64
	UMin     float64
	UMax     float64
	T        int
	N        int // Prediction horizon
	Q        float64
	R        float64
	Step     float64 // step size for control-theory candidates
}

func NewMPC() MPC {
	r := MPC{}

	return r
}

func (MPC) Update(setpoint float64) float64 {
	r := 0.0

	// Execute mpc update
	python := "/usr/bin/python3"
	mpcScript := "/mnt/c/Users/user/go/prog-conc-distribuida/sistemasadaptativos/controllers/mpc/mpc-script.py"
	cmd := exec.Command(python, mpcScript, "Go")
	out, err := cmd.CombinedOutput()
	shared.FailOnError(err, "Unable to execute Python script")

	// Check output
	sOut := strings.ReplaceAll(string(out), "\n", "")
	r, err = strconv.ParseFloat(sOut, 64)
	shared.FailOnError(err, "Unable to parse output")
	return r
}
