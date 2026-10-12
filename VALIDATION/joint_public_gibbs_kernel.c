/* Exact single-site conditionals for the fixed discrete-prior t5 model.
 * No RNG: caller supplies every uniform in deterministic sweep/site order.
 * The three physical components are retained for monitoring even in dep-only.
 */
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static double conditional(int lo, int hi, const double *r,
    const double *inverse, const double *av, const double *vav,
    const double *logprior, int likelihood_dims, double *prob) {
    double q=0., largest=-INFINITY, normalizer=0.;
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) q+=r[a]*inverse[3*a+b]*r[b];
    for(int j=lo;j<hi;j++) {
        double vq=q+vav[j];
        for(int a=0;a<3;a++) vq-=2*r[a]*av[3*j+a];
        if(vq<0.) vq=0.;
        prob[j-lo]=logprior[j]-.5*(5+likelihood_dims)*log1p(vq/5.);
        if(prob[j-lo]>largest) largest=prob[j-lo];
    }
    for(int j=lo;j<hi;j++) {prob[j-lo]=exp(prob[j-lo]-largest);normalizer+=prob[j-lo];}
    for(int j=lo;j<hi;j++) prob[j-lo]/=normalizer;
    return normalizer;
}

int gibbs_chain(int peers, int sweeps, int target, int likelihood_dims,
    const int64_t *offsets, const double *values, const double *vectors,
    const double *av, const double *vav, const double *logprior,
    const double *zero_delay, const double *observed_minus_remainder,
    const double *inverse, const double *uniforms, int64_t *states,
    double *trace, int64_t *changes) {
    int maxstates=0;
    for(int j=0;j<peers;j++) if(offsets[j+1]-offsets[j]>maxstates) maxstates=offsets[j+1]-offsets[j];
    double *prob=(double*)malloc(maxstates*sizeof(double));
    if(!prob) return 1;
    double total[3]={0.,0.,0.};
    for(int j=0;j<peers;j++) for(int a=0;a<3;a++) total[a]+=vectors[3*states[j]+a];
    for(int sweep=0;sweep<sweeps;sweep++) {
        for(int j=0;j<peers;j++) {
            int old=(int)states[j],lo=(int)offsets[j],hi=(int)offsets[j+1];
            double r[3];for(int a=0;a<3;a++) r[a]=observed_minus_remainder[a]-total[a]+vectors[3*old+a];
            double z=conditional(lo,hi,r,inverse,av,vav,logprior,likelihood_dims,prob);
            if(!isfinite(z)||z<=0.) {free(prob);return 2;}
            double u=uniforms[(int64_t)sweep*peers+j],cumulative=0.;int chosen=hi-1;
            for(int k=lo;k<hi;k++) {cumulative+=prob[k-lo];if(u<cumulative){chosen=k;break;}}
            states[j]=chosen;changes[j]+=(chosen!=old);
            for(int a=0;a<3;a++) total[a]+=vectors[3*chosen+a]-vectors[3*old+a];
        }
        /* Recompute to bound floating-point drift; this is not a model change. */
        for(int a=0;a<3;a++) total[a]=0.;
        double count=0.;for(int j=0;j<peers;j++) {
            for(int a=0;a<3;a++) total[a]+=vectors[3*states[j]+a];
            count+=zero_delay[states[j]];
        }
        int old=(int)states[target],lo=(int)offsets[target],hi=(int)offsets[target+1];
        double r[3];for(int a=0;a<3;a++) r[a]=observed_minus_remainder[a]-total[a]+vectors[3*old+a];
        double z=conditional(lo,hi,r,inverse,av,vav,logprior,likelihood_dims,prob);
        if(!isfinite(z)||z<=0.) {free(prob);return 3;}
        double mean=0.,copy=0.;for(int k=lo;k<hi;k++){mean+=prob[k-lo]*values[k];copy+=prob[k-lo]*zero_delay[k];}
        double *out=trace+8*sweep;out[0]=mean;out[1]=copy;out[2]=values[old];
        for(int a=0;a<3;a++) out[3+a]=total[a];out[6]=count;
        double q=0.;for(int a=0;a<3;a++) for(int b=0;b<3;b++)
            q+=(observed_minus_remainder[a]-total[a])*inverse[3*a+b]*(observed_minus_remainder[b]-total[b]);
        out[7]=-.5*(5+likelihood_dims)*log1p(fmax(q,0.)/5.);
    }
    free(prob);return 0;
}
