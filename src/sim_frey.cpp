/*
 * Copyright 2014-2016 Friedemann Zenke, Lorric Ziegler
 *
 * This file is part of Auryn, a simulation package for plastic
 * spiking neural networks.
 *
 * Auryn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Auryn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
 */


#include "auryn.h"
#include "ZynapseConnection.h"
#include "ZynapseMonitor.h"
#include "PRPGroup.h"

#include <sys/time.h>

namespace po = boost::program_options;

using namespace auryn;

bool before(char *i, char *j) {
        std::stringstream ssi(i);
        std::stringstream ssj(j);
        AurynFloat ti, tj;
        ssi >> ti;
        ssj >> tj;
        return ti<tj;
}

void write_train(std::ofstream *outfile, char **chTimes, int ntimes) {
        std::sort(chTimes, chTimes+ntimes, before);
        for (int i = 0; i < ntimes; i++) {
                char buffer[32];
                int n = sprintf(buffer, "%s\n", chTimes[i]);
                outfile->write(buffer, n);
        }
}

int generate_raster(char *filename, int prot, int n_neuron, Logger *logger) {

        int *n_pulse = new int[4] {21, 100, 1, 3};
        int *n_train = new int[4] {1, 1, 900, 900};
        AurynFloat *dt_pulse = new AurynFloat[4] {0.01, 0.01, 0., 0.05};
        AurynFloat *dt_train = new AurynFloat[4] {0., 0., 1., 1.};

        boost::mt19937 gen;
        boost::normal_distribution<> dist(0., 0.003);
        boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
                die(gen, dist);

        timeval s;
        gettimeofday(&s, NULL);
        int n_seed = s.tv_sec;
        gen.seed(n_seed);

        AurynFloat offset = 1; // to avoid negative times

        int ntimes = n_pulse[prot]*n_neuron;
        char *chTimes[ntimes];
        for (int i = 0; i < ntimes; i++)
                chTimes[i] = new char[32];

        std::ofstream outfile;
        outfile.open(filename, std::ios::out);
        if (!outfile) {
                std::stringstream oss;
                oss << "Can't open output file " << filename;
                logger->msg(oss.str(),ERROR);
                return 1;
        }
        outfile.setf(std::ios::fixed);
        outfile.precision(4);

        for (int nt = 0; nt < n_train[prot]; nt++) {
                for (int np = 0; np < n_pulse[prot]; np++) {
                        for (int nn = 0; nn < n_neuron; nn++) {
                                char *ti = chTimes[np*n_neuron+nn];
                                AurynFloat t = die()+offset;
                                sprintf(ti, "%f %d", t, nn);
                        }
                        offset += dt_pulse[prot];
                }
                write_train(&outfile, chTimes, ntimes);
                offset += dt_train[prot] - n_pulse[prot]*dt_pulse[prot];
        }

        outfile.close();

        return 0;
}

int main(int ac, char* av[])
{
        std::string dir = ".";
        const char * file_prefix = "frey";

        char strbuf [255];
        std::string msg;

        bool verbose = false;

        int n_in = 2000;
        int n_out = 10;

        double sparseness = 0.1;
        double weight = 0.05;
        double am = 2e-4;
        double ap = 5e-4;

        double noise = 1e-4;
        double tau = 200.;

        double pretime = 0.;
        double time = 0.25;
        double posttime = 3600.;

        AurynFloat monitor_time = 60.;

        string protocol = "wtet";
        int prot = 0;
        bool dopamine = false;

        double zup = 0.33;

        int n_rec = 10;

        int errcode = 0;

        try {

                po::options_description desc("Allowed options");
                desc.add_options()
                        ("help,h", "produce help message")
                        ("verbose,v", "verbose mode")
                        ("pre", po::value<double>(), "pre time [0.]")
                        ("tot,t", po::value<double>(), "total time [3600.]")
                        ("sparseness,s", po::value<double>(), "sparseness [0.1]")
                        ("protocol,p", po::value<int>(), "protocol 0-3 [WTET,stet,wlfs,slfs]")
                        ("nrec,n", po::value<int>(), "number of recorded synapses [10]")
                        ("weight,w", po::value<double>(), "weight [0.05]")
                        ("am", po::value<double>(), "depression cte [2e-4]")
                        ("ap", po::value<double>(), "potentiation cte [5e-4]")
                        ("noise", po::value<double>(), "noise [1e-4]")
                        ("tau", po::value<double>(), "synaptic time cte [200]")
                        ("montime,m", po::value<double>(), "monitor record interval [60.]")
                        ("dir", po::value<string>(), "output dir")
                        ;

                po::variables_map vm;
                po::store(po::parse_command_line(ac, av, desc), vm);
                po::notify(vm);

                if (vm.count("help")) {
                        std::cout << desc << "\n";
                        return 1;
                }

                if (vm.count("verbose")) {
                        verbose = true;
                }

                if (vm.count("pre")) {
                        std::cout << "pre time set to "
                                  << vm["pre"].as<double>() << ".\n";
                        pretime = vm["pre"].as<double>();
                }

                if (vm.count("tot")) {
                        std::cout << "total time set to "
                                  << vm["tot"].as<double>() << ".\n";
                        posttime = vm["tot"].as<double>();
                }

                if (vm.count("sparseness")) {
                        std::cout << "sparseness set to "
                                  << vm["sparseness"].as<double>() << ".\n";
                        sparseness = vm["sparseness"].as<double>();
                }

                if (vm.count("protocol")) {
                        prot = vm["protocol"].as<int>();
                        std::cout << "protocol set to ";
                        switch (prot) {
                        case 0:
                                std::cout << "wtet.\n";
                                protocol = "wtet";
                                time = 0.25;
                                break;
                        case 1:
                                std::cout << "stet.\n";
                                protocol = "stet";
                                time = 1201.;
                                dopamine = true;
                                break;
                        case 2:
                                std::cout << "wlfs.\n";
                                protocol = "wlfs";
                                time = 900.;
                                break;
                        case 3:
                                std::cout << "slfs.\n";
                                protocol = "slfs";
                                time = 900.;
                                dopamine = true;
                                break;
                        }
                }

                if (vm.count("nrec")) {
                        n_rec = vm["nrec"].as<int>();
                        std::cout << "n_rec set to " << n_rec << ".\n";
                }

                if (vm.count("weight")) {
                        std::cout << "weight set to "
                                  << vm["weight"].as<double>() << ".\n";
                        weight = vm["weight"].as<double>();
                }

                if (vm.count("am")) {
                        std::cout << "am set to "
                                  << vm["am"].as<double>() << ".\n";
                        am = vm["am"].as<double>();
                }

                if (vm.count("ap")) {
                        std::cout << "ap set to "
                                  << vm["ap"].as<double>() << ".\n";
                        ap = vm["ap"].as<double>();
                }

                if (vm.count("noise")) {
                        std::cout << "noise set to "
                                  << vm["noise"].as<double>() << ".\n";
                        noise = vm["noise"].as<double>();
                }

                if (vm.count("tau")) {
                        std::cout << "tau set to "
                                  << vm["tau"].as<double>() << ".\n";
                        tau = vm["tau"].as<double>();
                }

                if (vm.count("montime")) {
                        std::cout << "monitor_time set to "
                                  << vm["montime"].as<double>() << ".\n";
                        monitor_time = vm["montime"].as<double>();
                }

                if (vm.count("dir")) {
                        std::cout << "dir set to "
                                  << vm["dir"].as<string>() << ".\n";
                        dir = vm["dir"].as<string>();
                }

        }
        catch(std::exception& e) {
                std::cerr << "error: " << e.what() << "\n";
                return 1;
        }
        catch(...) {
                std::cerr << "Exception of unknown type!\n";
        }

		auryn_init(ac, av);

        LogMessageType log_level_file = PROGRESS;
        if ( verbose ) log_level_file = EVERYTHING;
		logger->set_logfile_loglevel(log_level_file);

        msg =  "Generating raster ...";
        logger->msg(msg,PROGRESS,true);

        sprintf(strbuf, "%s/%s_%s.ras", dir.c_str(), file_prefix, protocol.c_str());
        errcode = generate_raster(strbuf, prot, n_in, logger);

        msg =  "Setting up neuron groups ...";
        logger->msg(msg,PROGRESS,true);

        FileInputGroup * tetanus = new FileInputGroup(n_in, strbuf, false);

        PRPGroup * neuron = new PRPGroup(n_out);

        msg =  "Setting up E connections ...";
        logger->msg(msg,PROGRESS,true);

        ZynapseConnection *con = \
                new ZynapseConnection(tetanus, neuron, weight, sparseness, am, ap);
        con->random_data_potentiation(zup);
        con->set_noise(noise);
        con->set_tau(tau,0);con->set_tau(tau,1);con->set_tau(tau,2);

        msg = "Setting up monitors ...";
        logger->msg(msg,PROGRESS,true);

        sprintf(strbuf, "%s/%s_%s_x.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
        WeightStatsMonitor * wsmon_x = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 0);
        sprintf(strbuf, "%s/%s_%s_y.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
        WeightStatsMonitor * wsmon_y = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 1);
        sprintf(strbuf, "%s/%s_%s_z.%d.wgs", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
        WeightStatsMonitor * wsmon_z = \
                new WeightStatsMonitor(con, strbuf, monitor_time, 2);

        // to count states (directly on data files): if x_i>0 s+=2^i
        if (n_rec>0) {
                sprintf(strbuf, "%s/%s_%s_x.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
                WeightMonitor * xmon =                                  \
                        new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 0);
                sprintf(strbuf, "%s/%s_%s_y.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
                WeightMonitor * ymon =                                  \
                        new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 1);
                sprintf(strbuf, "%s/%s_%s_z.%d.syn", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
                WeightMonitor * zmon =                                  \
                        new WeightMonitor(con, 0, n_rec, strbuf, monitor_time, DATARANGE, 2);
        }

        sprintf(strbuf, "%s/%s_%s.%d.ras", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
        SpikeMonitor * smon = new SpikeMonitor(neuron, strbuf);

        sprintf(strbuf, "%s/%s_%s.%d.zyn", dir.c_str(), file_prefix, protocol.c_str(), sys->mpi_rank());
        ZynapseMonitor * zmon = new ZynapseMonitor(con, strbuf, monitor_time);

        msg = "Simulating ...";
        logger->msg(msg,PROGRESS,true);

        // pre
        tetanus->active = false;
        if (!sys->run(pretime, false) )
                errcode = 1;
        // stimulus
        tetanus->active = true;
        if (!sys->run(time, false) )
                errcode = 1;
        if (dopamine) {
                neuron->dopamine_on();
                if (!sys->run(60, false) )
                        errcode = 1;
                neuron->dopamine_off();
                posttime -= 60;
        }
        // post
        double remaining = posttime - time;
        if (!sys->run(remaining, false) )
                errcode = 1;

        if (errcode)
			auryn_abort(errcode);

        msg = "Freeing ...";
		auryn_free();

        return errcode;
}
