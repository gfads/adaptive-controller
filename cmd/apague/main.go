package main

import (
	"control/internal/shared"
	"fmt"
	"os"
	"reflect"
)

func main() {
	os.Setenv("ControlDir", "/mnt/d/GolandProjects/control-theory")
	os.Setenv("ExpConf_Setpoints", "[10 10 10 10]")

	conf := shared.LoadConfig(shared.Publisher)
	temp := shared.EnvToConfig("ExpConf_Setpoints", reflect.TypeOf(conf.Experiment.Setpoints)).([]float64)
	fmt.Println(temp)
	fmt.Println("Hello")
}
