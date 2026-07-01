package main

import (
	"control/internal/shared"
	"os"
	"os/exec"
)

func main() {

	if len(os.Args) <= 1 {
		shared.ErrorHandler(shared.GetFunction(), "Not enough OS arguments")
	}
	caller := os.Args[1:][0]
	if caller != shared.Subscriber && caller != shared.Publisher {
		shared.ErrorHandler(shared.GetFunction(), "Wrong OS arguments. It should be '"+shared.Subscriber+"' or '"+shared.Publisher+"'")
	}

	// update environment variables of ~/.bashrc
	cmd := exec.Command("bash", "-c", "source ~/.bashrc")
	_, err := cmd.Output()
	if err != nil {
		shared.ErrorHandler(shared.GetFunction(), err.Error())
	}

	// generate env file
	fileContent := shared.GenerateEnvFile(caller)
	shared.SetEnvVariables(fileContent)
}
