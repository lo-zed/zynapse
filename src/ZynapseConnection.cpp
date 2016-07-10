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

#include "ZynapseConnection.h"

using namespace auryn;

boost::mt19937 ZynapseConnection::zynapse_connection_gen = boost::mt19937();
bool ZynapseConnection::has_been_seeded = false;

/********************
 *** constructors ***
 ********************/

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     TransmitterType transmitter)
        : DuplexConnection(source, destination, transmitter)

{
        init(1, KW, AM, AP);
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     AurynFloat wo, AurynFloat sparseness,
                                     TransmitterType transmitter)
        : DuplexConnection(source, destination, wo, sparseness,
                           transmitter, "ZynapseConnection")

{
        init(wo, KW, AM, AP);
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     AurynFloat wo, AurynFloat sparseness,
                                     AurynFloat a_m, AurynFloat a_p, AurynFloat kw,
                                     TransmitterType transmitter, string name)
        : DuplexConnection(source, destination, wo, sparseness,
                           transmitter, name)

{
        init(wo, kw, a_m, a_p);
        if ( name.empty() )
                set_name("ZynapseConnection");
}

ZynapseConnection::ZynapseConnection(SpikingGroup *source, NeuronGroup *destination,
                                     const char *filename, AurynFloat wo,
                                     AurynFloat a_m, AurynFloat a_p, AurynFloat kw,
                                     TransmitterType transmitter)
        : DuplexConnection(source, destination, filename, transmitter)
{
        init(wo, kw, a_m, a_p);
}

/*****************
 *** init crap ***
 *****************/

ZynapseConnection::~ZynapseConnection()
{
        if (dst->get_post_size() > 0)
                free();
}

void ZynapseConnection::free()
{
        delete dist;
        delete die;
        delete tr_gxy;
}

void ZynapseConnection::init(AurynFloat wo, AurynFloat k_w, AurynFloat a_m, AurynFloat a_p)
{
        if (dst->get_post_size() == 0) return;

        dist = new boost::normal_distribution<> (0., 1.);
        die = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
                (zynapse_connection_gen, *dist);
        if (!has_been_seeded)
                seed(12345*auryn::communicator->rank());

        set_min_weight(wo);
        set_max_weight(k_w*wo);

        tr_pre = src->get_pre_trace(TAU_PRE);
        tr_post = dst->get_post_trace(TAU_POST);
        tr_long = dst->get_post_trace(TAU_LONG);

        set_plast_constants(a_m, a_p);
        stdp_active = true;

        euler[0] = TUPD/TAUX;
        euler[1] = TUPD/TAUY;
        euler[2] = TUPD/TAUZ;

        coeff[0] = 4/wo/wo/(k_w-1)/(k_w-1);
        coeff[1] = wo*wo*wo*k_w*(1+k_w)/2;
        coeff[2] = wo*wo*(1+k_w)*(1+k_w)/2 + wo*wo*k_w;
        coeff[3] = 3*wo*(1+k_w)/2;

        timestep_synapses = TUPD/dt;

        eta = wo*(k_w-1)*sqrt(ETAXYZ*TUPD)/2;

        // Set number of synaptic states
        w->set_num_synapse_states(3);

        // copy all the elements from z=0 to z=1,2
        w->state_set_all(w->get_state_begin(1),0.0);
        w->state_set_all(w->get_state_begin(2),0.0);
        w->state_add(w->get_state_begin(0),w->get_state_begin(1));
        w->state_add(w->get_state_begin(0),w->get_state_begin(2));

        tr_gxy = new LinearTrace(get_nonzero(), TAUG, sys->get_clock_ptr());

        // Run finalize again to rebuild backward matrix
        finalize();
}

void ZynapseConnection::set_plast_constants(AurynFloat a_m, AurynFloat a_p)
{
        am = a_m/TAU_POST;
        ap = a_p/TAU_PRE/TAU_LONG;
}

void ZynapseConnection::finalize() {
        DuplexConnection::finalize();
        init_shortcuts();
}

void ZynapseConnection::init_shortcuts()
{
        if ( dst->get_post_size() == 0 ) return; // if there are no target neurons on this rank

        fwd_ind = w->get_row_begin(0);
        fwd_data = w->get_data_begin();

        bkw_ind = bkw->get_row_begin(0);
        bkw_data = bkw->get_data_begin();
}

/************
 *** body ***
 ************/

void ZynapseConnection::integrate()
{
        AurynWeight *x = w->get_state_begin(0),
                *y = w->get_state_begin(1),
                *z = w->get_state_begin(2);

        AurynWeight prot = *dst->get_state_variable("prp");
        for (AurynLong i = 0 ; i < w->get_nonzero() ; ++i ) {
                AurynWeight xyi = x[i] - y[i],
                        yzi = y[i] - z[i];
                AurynInt gxy;
                if (tr_gxy->get(i)>THETAG) gxy = 1;
                else gxy = 0;
                x[i] += euler[0]*(coeff[0]*(coeff[1]-x[i]*(coeff[2]-x[i]*(coeff[3]-x[i]) ) ) -
                                  META_YX*(1-gxy)*xyi
                                  ) + eta*(*die)();
                y[i] += euler[1]*(coeff[0]*(coeff[1]-y[i]*(coeff[2]-y[i]*(coeff[3]-y[i]) ) ) +
                                  TILT*gxy*xyi -
                                  META_ZY*(1-prot)*yzi
                                  ) + eta*(*die)();
                z[i] += euler[2]*(coeff[0]*(coeff[1]-z[i]*(coeff[2]-z[i]*(coeff[3]-z[i]) ) ) +
                                  TILT*prot*yzi
                                  ) + eta*(*die)();
        }
}

void ZynapseConnection::propagate_forward()
{
        // loop over all spikes (yields presynaptic cell ids of cells that spiked)
        for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
             spike != src->get_spikes()->end() ; ++spike ) {
                // loop over all postsynaptic partners the cells
                // that are targeted by that presynaptic cell
                for (const NeuronID * c = w->get_row_begin(*spike) ;
                     c != w->get_row_end(*spike) ;
                     ++c ) { // c = post index

                        // determines the weight of connection
                        AurynWeight * weight = w->get_data_ptr(c);
                        // evokes the postsynaptic response
                        transmit( *c , *weight );

                        // handles plasticity
                        if ( stdp_active ) {

                                // performs weight update upon presynaptic spike
                                dw_pre(c, weight);

                        }
                }
        }
}

void ZynapseConnection::propagate_backward()
{
        if (stdp_active) {
                SpikeContainer::const_iterator spikes_end = dst->get_spikes_immediate()->end();
                // loop over all spikes
                for (SpikeContainer::const_iterator spike = dst->get_spikes_immediate()->begin();
                     spike != spikes_end; // spike = post_spike
                     ++spike ) {
                        // Since we need the local id of the postsynaptic neuron that spiked
                        // multiple times, we translate it here:
                        NeuronID translated_spike = dst->global2rank(*spike);

                        // loop over all presynaptic partners
                        for (const NeuronID * c = bkw->get_row_begin(*spike);
                             c != bkw->get_row_end(*spike); ++c ) {

#ifdef CODE_ACTIVATE_PREFETCHING_INTRINSICS
                                // prefetches next memory cells to reduce number of
                                // last-level cache misses
                                _mm_prefetch((const char *)bkw_data[c-bkw_ind+2],  _MM_HINT_NTA);
#endif

                                // computes plasticity update
                                AurynWeight * weight = bkw->get_data(c);
                                dw_post(c,translated_spike,weight);
                        }
                }
        }
}

/*! This function implements what happens to synapes transmitting a
 *  spike to neuron 'post'. */
void ZynapseConnection::dw_pre(const NeuronID * post, AurynWeight * weight)
{
        // translate post id to local id on rank: translated_spike
        NeuronID translated_spike = dst->global2rank(*post),
                data_ind = post-fwd_ind;
        // NOTE get_data(data_ind) = get_data(post) !
        AurynDouble dw = am*tr_post->get(translated_spike),
                reset = *weight-w->get_data(post,2);
        if (reset>0) dw *= 1+C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset<0) {
                AurynDouble gxy = tr_gxy->get(data_ind);
                tr_gxy->add(data_ind, dw*(1.-gxy));
        }
        dw *= wmin-*weight;
        *weight += dw;
}

/*! This function implements what happens to synapes experiencing a
 *  backpropagating action potential from neuron 'pre'. */
void ZynapseConnection::dw_post(const NeuronID * pre, NeuronID post, AurynWeight * weight)
{
        // at this point post was already translated to a local id in
        // the propagate_backward function below.
        NeuronID data_ind = bkw_data[pre-bkw_ind]-fwd_data;
        AurynDouble dw = ap*tr_pre->get(*pre)*tr_long->get(post),
                reset = w->get_data(data_ind,2)-*weight;
        if (reset>0) dw *= 1+C_RESET*reset;
        if (dw>1) dw = 1;
        if (reset<0) {
                AurynDouble gxy = tr_gxy->get(data_ind);
                tr_gxy->add(data_ind, dw*(1.-gxy));
        }
        dw *= wmax-*weight;
        *weight += dw;
}

void ZynapseConnection::propagate()
{
        propagate_forward();
        propagate_backward();
}

void ZynapseConnection::evolve()
{
        if (dst->get_post_size() > 0 && sys->get_clock()%timestep_synapses==0)
                integrate();
}

void ZynapseConnection::random_data_potentiation(AurynFloat z_up, bool reset)
{
        if (reset) {
                depress();
        }
        if (z_up) {
                boost::exponential_distribution<> exp_dist(z_up);
                boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> >
                        exp_die(zynapse_connection_gen, exp_dist);

                AurynLong x = (AurynLong) exp_die(), y;
                while (x < get_nonzero() ) {
                        potentiate(x);
                        y = (AurynLong)(exp_die()+0.5);
                        x += (y==0)?1:y;
                }
        }
}

void ZynapseConnection::seed(int s)
{
        std::stringstream oss;
        oss << get_log_name() << "Seeding with " << s;
        auryn::logger->msg(oss.str(),VERBOSE);
        zynapse_connection_gen.seed(s);
        has_been_seeded = true;
}

void ZynapseConnection::potentiate(NeuronID i)
{
        for (int z=0; z<3; z++)
                w->set_data(i,wmax,z);
}

void ZynapseConnection::potentiate()
{
        for (int z=0; z<3; z++) {
                w->state_set_all(w->get_state_begin(z),wmax);
        }
}

void ZynapseConnection::depress()
{
        for (int z=0; z<3; z++) {
                w->state_set_all(w->get_state_begin(z),wmin);
        }
}

void ZynapseConnection::set_noise(AurynFloat level)
{
        eta = (wmax-wmin)*sqrt(level*TUPD)/2;
}

void ZynapseConnection::set_tau(AurynFloat level, NeuronID z)
{
        euler[z] = TUPD/level;
}

AurynFloat ZynapseConnection::get_g(NeuronID i)
{
        return tr_gxy->get(i);
}

AurynFloat ZynapseConnection::get_prp()
{
        return *dst->get_state_variable("prp");
}

void ZynapseConnection::g_stats(AurynDouble &mean, AurynDouble &std)
{
        double sum = 0; // needs double here -- machine precision really matters here
        double sum2 = 0;

        NeuronID count = get_nonzero();

        for ( AurynWeight i = 0 ; i != count ; ++i ) {
                sum  += get_g(i);
                sum2 += (get_g(i) * get_g(i));
        }

        if ( count <= 1 ) {
                mean = sum;
                std = 0;
                return;
        }

        mean = sum/count;
        std = sqrt(sum2/count-mean*mean);
}
