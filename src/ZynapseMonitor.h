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
 *
 * If you are using Auryn or parts of it for your work please cite:
 * Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations
 * of spiking neural networks using general-purpose computers.
 * Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
 */

#ifndef ZYNAPSEMONITOR_H_
#define ZYNAPSEMONITOR_H_

#include "auryn_definitions.h"
#include "AurynVector.h"
#include "Monitor.h"
#include "System.h"
#include "ZynapseConnection.h"
#include <fstream>
#include <iomanip>

namespace auryn {

        /*! \brief Records protein level of target as well as mean and standard deviation
	 * of g_trace of ZynapseConnection in predefined intervals.
         */
        class ZynapseMonitor : protected Monitor
        {
        protected:
                ZynapseConnection * src;
                AurynTime ssize;
                void init(ZynapseConnection * source, string filename, AurynTime stepsize);

        public:
                ZynapseMonitor(ZynapseConnection * source, string filename, AurynDouble binsize=1.0);
                virtual ~ZynapseMonitor();
                void propagate();
        };

}

#endif /*ZYNAPSEMONITOR_H_*/
