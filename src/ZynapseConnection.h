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

#ifndef ZYNAPSECONNECTION_H_
#define ZYNAPSECONNECTION_H_

#include "auryn_definitions.h"
#include "DuplexConnection.h"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/variate_generator.hpp>

/*! constants for the synapse model
 */
#define TILT 3.5*0.25
#define META_YX 1.3*0.25
#define META_ZY 0.95*0.25

#define TAUX 200.
#define TAUY 200.
#define TAUZ 200.

#define TUPD 100e-3

#define TAUG 600.

#define ETAXYZ 0.0001

#define THETAG 0.37 // e^-1

#define KW 3 // 3 for frey

/*! constants for the plasticity rule
 */
#define AM 1e-3
#define AP 1e-3
#define TAU_PRE 0.0168
#define TAU_POST 0.0337
#define TAU_LONG 0.04
#define C_RESET 1.

namespace auryn {

        /*! \brief Implements complex synapse as described by Ziegler et al. 2015.
         */
        class ZynapseConnection : public DuplexConnection
        {
        private:

                void virtual_serialize(boost::archive::binary_oarchive & ar, const unsigned int version )
                {
                        DuplexConnection::virtual_serialize(ar,version);
                }

                void virtual_serialize(boost::archive::binary_iarchive & ar, const unsigned int version )
                {
                        DuplexConnection::virtual_serialize(ar,version);
                        DuplexConnection::compute_reverse_matrix(); // just in case the buffer location has changed
                }

                AurynFloat euler[3], coeff[4], eta, ap, am;
                int timestep_synapses;

                void init(AurynFloat wo, AurynFloat k_w, AurynFloat a_m,
                          AurynFloat a_p);
                void init_shortcuts();

                static boost::mt19937 zynapse_connection_gen;
                boost::normal_distribution<> *dist;
                boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > * die;

                static bool has_been_seeded;

                void free();
                virtual void finalize();

        protected:

                NeuronID * fwd_ind;
                AurynWeight * fwd_data;

                NeuronID * bkw_ind;
                AurynWeight ** bkw_data;

                /* Definitions of presynaptic traces */
                PRE_TRACE_MODEL * tr_pre;

                /* Definitions of postsynaptic traces */
                DEFAULT_TRACE_MODEL * tr_post;
                DEFAULT_TRACE_MODEL * tr_long;

                /*! This function propagates spikes from pre to postsynaptic cells
                 * and performs plasticity updates upon presynaptic spikes. */
                void propagate_forward();

                /*! This performs plasticity updates following postsynaptic spikes.
                 * To that end the postsynaptic spikes have to be communicated backward
                 * to the corresponding synapses connecting to presynaptic neurons. This
                 * is why this function is called propagate_backward ... it is remeniscent
                 * of a back-propagating action potential. */
                void propagate_backward();

                /*! Action on weight upon presynaptic spike on connection with postsynaptic
                 * partner post. This function should be modified to define new spike based
                 * plasticity rules.
                 * @param post the postsynaptic cell from which the synaptic trace is read out
                 * @param weight the synaptic weight to be depressed
                 */
                void dw_pre(const NeuronID * post, AurynWeight * weight);

                /*! Action on weight upon postsynaptic spike of cell post on connection
                 * with presynaptic partner pre. This function should be modified to define
                 * new spike based plasticity rules.
                 * @param pre the presynaptic cell in question.
                 * @param post the postsynaptic cell in question.
                 * @param weight the synaptic weight to be potentiated
                 */
                void dw_post(const NeuronID * pre, NeuronID post, AurynWeight * weight);

                void integrate();

                LinearTrace *tr_gxy;

        public:

                ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                  TransmitterType transmitter=GLUT);
                ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                  AurynFloat wo, AurynFloat sparseness, TransmitterType transmitter=GLUT);
                /*! Default constructor. Sets up a random sparse connection and plasticity parameters
                 *
                 * @param source the presynaptic neurons.
                 * @param destination the postsynaptic neurons.
                 * @param wo the initial synaptic weight and lower fixed point of weight dynamics.
                 * @param sparseness the sparseness of the connection (probability of connection).
                 * @param a_m the depression learning rate.
                 * @param a_p the potentiation learning rate.
                 * @param kw the ratio high/low weight (default is 3).
                 * @param transmitter the TransmitterType (default is GLUT, glutamatergic).
                 * @param name a sensible identifier for the connection used in debug output.
                 */
                ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                  AurynFloat wo, AurynFloat sparseness,
                                  AurynFloat a_m, AurynFloat a_p, AurynFloat kw=KW,
                                  TransmitterType transmitter=GLUT,
                                  string name = "ZynapseConnection");
                ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                  const char * filename, AurynFloat wo, AurynFloat a_m,
                                  AurynFloat a_p, AurynFloat kw=KW,
                                  TransmitterType transmitter=GLUT);

                virtual ~ZynapseConnection();

                virtual void evolve();
                virtual void propagate();

                /*! Toggle stdp active/inactive. When inactive traces are still updated,
                 * but weights are not.
                 */
                bool stdp_active;

                void random_data_potentiation(AurynFloat z_up, bool reset=false);

                void seed(int s);

                void set_plast_constants(AurynFloat a_m, AurynFloat a_p);
                void potentiate(NeuronID i);
                void potentiate();
                void depress();

                void set_noise(AurynFloat level);
                void set_tau(AurynFloat level, NeuronID z);

                AurynFloat get_g(NeuronID i);
                AurynFloat get_prp();
                void g_stats(AurynDouble &mean, AurynDouble &std);
        };

}

#endif /*ZYNAPSECONNECTION_H_*/
