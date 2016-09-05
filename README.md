# Auryn classes and simulation to reproduce slice experiments on synaptic consolidation.

These simulations are to reproduce the findings from Ziegler, L., Zenke, F., Kastner, D.B. and Gerstner, W. (2015). 
Synaptic consolidation: from synapses to behavioral modeling. JNeurosci 35.3
http://www.jneurosci.org/content/35/3/1319.full

## Auryn classes

```
ZynapseConnection
	Implements the excitatory connection object for complex
	synaptic plasticity with protein-dependent consolidation.

ZynapseMonitor
	Monitors the plasticity-related proteins (PRP) level
	as well as the mean and standard deviation of g(t)
	(the synapse specific tag-gating trace).

PRPGroup
	Implements an Adaptive Integrate and Fire neuron group
	with PRPs.
```

## Running an example

To run slice experiment simulation as described in Figure 3A of Ziegler et al. (2015),
you need to install and compile Auryn v0.8.0-beta3. 
In the following we will assume that you have git installed and up and running
on your system. Moreover, you have all dependencies to compile Auryn installed.

### Download and compile Auryn 

In essence do the following:
```
$ cd ~
$ git clone https://github.com/fzenke/auryn.git
$ cd auryn
$ git checkout -b v0.8.0-beta3 v0.8.0-beta3
$ cd build/release
$ make
```

Should you have difficulties compiling the simulator please refer to the
installation and troubleshooting section in the manual (www.fzenke.net/auryn).
If you had auryn already installed and have problems checking out version 0.8.0 beta3
run `git fetch --tags`.

### Compile simulation classes and programs

Now go to the installation directory of the `src/` simulation code (when you are
reading this, chances are you are already in this directory). And run `make`
there (if you cloned the repository in some other place than `~`, edit the first line of
the Makefile accordingly). This should build the necessary Auryn libraries that implement
plasticity and the simulation libraries. For instance the simulation program `sim_frey` is the
one behind Figure 3A in Ziegler et al. (2015).

### Run simulation

Invoking `$ ./sim_frey`, or `$ mpirun -n 2 sim_frey` for example,
will generate an input raster and run the simulation of a weak tetanus (WTET) with a total
time of one hour. For other options run a simulation with the flag `-h` or `--help`.

### Plot the result

Assuming you have gnuplot installed on your system, go to the directory where you saved the results
(editable with the flag `--dir` in sim_frey) and run this line in gnuplot
`plot "frey_wtet_x.1.wgs" w l t 'weight', "frey_wtet_y.1.wgs" w l t 'tag', "frey_wtet_z.1.wgs" w l t 'scaffold'`
(this is if you ran the simulation on two nodes, else change `.1` accordingly).