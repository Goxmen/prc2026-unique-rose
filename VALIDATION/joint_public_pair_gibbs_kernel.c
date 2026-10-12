/* Exact target/site block conditionals for the unchanged discrete t5 model.
 * The target is jointly refreshed with each other site in a fixed sweep.
 * Caller supplies all uniforms. No likelihood, support or prior is changed.
 */
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

static double one(int lo, int hi, const double *r, const double *inverse,
    const double *av, const double *vav, const double *logprior, int dims, double *prob) {
    double q=0., largest=-INFINITY, z=0.;
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) q+=r[a]*inverse[3*a+b]*r[b];
    for(int j=lo;j<hi;j++) {
        double vq=q+vav[j];
        for(int a=0;a<3;a++) vq-=2*r[a]*av[3*j+a];
        prob[j-lo]=logprior[j]-.5*(5+dims)*log1p(fmax(vq,0.)/5.);
        if(prob[j-lo]>largest) largest=prob[j-lo];
    }
    for(int j=0;j<hi-lo;j++) {prob[j]=exp(prob[j]-largest);z+=prob[j];}
    for(int j=0;j<hi-lo;j++) prob[j]/=z;
    return z;
}

static double pair(int lo, int hi, int tl, int th, const double *r,
    const double *inverse, const double *vectors, const double *av,
    const double *vav, const double *logprior, int dims, double *prob) {
    double q=0., largest=-INFINITY, z=0.;int n=0;
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) q+=r[a]*inverse[3*a+b]*r[b];
    for(int j=lo;j<hi;j++) for(int t=tl;t<th;t++) {
        double vq=q+vav[j]+vav[t];
        for(int a=0;a<3;a++) vq+=2*vectors[3*j+a]*av[3*t+a]-2*r[a]*(av[3*j+a]+av[3*t+a]);
        prob[n]=logprior[j]+logprior[t]-.5*(5+dims)*log1p(fmax(vq,0.)/5.);
        if(prob[n]>largest) largest=prob[n];
        n++;
    }
    for(int j=0;j<n;j++) {prob[j]=exp(prob[j]-largest);z+=prob[j];}
    for(int j=0;j<n;j++) prob[j]/=z;
    return z;
}

int gibbs_chain(int peers, int sweeps, int target, int dims,
    const int64_t *offsets, const double *values, const double *vectors,
    const double *av, const double *vav, const double *logprior,
    const double *zero_delay, const double *observed, const double *inverse,
    const double *uniforms, int64_t *states, double *trace, int64_t *changes) {
    int largest=0,tl=(int)offsets[target],th=(int)offsets[target+1];
    for(int j=0;j<peers;j++) if(offsets[j+1]-offsets[j]>largest) largest=(int)(offsets[j+1]-offsets[j]);
    double *prob=(double*)malloc((size_t)largest*(th-tl)*sizeof(double));
    if(!prob) return 1;
    double total[3]={0.,0.,0.};
    for(int j=0;j<peers;j++) for(int a=0;a<3;a++) total[a]+=vectors[3*states[j]+a];
    for(int sweep=0;sweep<sweeps;sweep++) {
        for(int j=0;j<peers;j++) {
            int old=(int)states[j], ot=(int)states[target];
            int lo=(int)offsets[j],hi=(int)offsets[j+1],n=hi-lo;
            double r[3];
            for(int a=0;a<3;a++) r[a]=observed[a]-total[a]+vectors[3*old+a]+(j==target?0.:vectors[3*ot+a]);
            double z;
            if(j==target) z=one(lo,hi,r,inverse,av,vav,logprior,dims,prob);
            else {n*=th-tl;z=pair(lo,hi,tl,th,r,inverse,vectors,av,vav,logprior,dims,prob);}
            if(!isfinite(z)||z<=0.) {free(prob);return 2;}
            double u=uniforms[(int64_t)sweep*peers+j],cumulative=0.;int chosen=n-1;
            for(int k=0;k<n;k++) {cumulative+=prob[k];if(u<cumulative){chosen=k;break;}}
            int ns=lo+(j==target?chosen:chosen/(th-tl));
            states[j]=ns;changes[j]+=(ns!=old);
            for(int a=0;a<3;a++) total[a]+=vectors[3*ns+a]-vectors[3*old+a];
            if(j!=target) {
                int nt=tl+chosen%(th-tl);states[target]=nt;changes[target]+=(nt!=ot);
                for(int a=0;a<3;a++) total[a]+=vectors[3*nt+a]-vectors[3*ot+a];
            }
        }
        for(int a=0;a<3;a++) total[a]=0.;
        double count=0.;
        for(int j=0;j<peers;j++) {
            for(int a=0;a<3;a++) total[a]+=vectors[3*states[j]+a];
            count+=zero_delay[states[j]];
        }
        int old=(int)states[target];double r[3];
        for(int a=0;a<3;a++) r[a]=observed[a]-total[a]+vectors[3*old+a];
        double z=one(tl,th,r,inverse,av,vav,logprior,dims,prob);
        if(!isfinite(z)||z<=0.) {free(prob);return 3;}
        double mean=0.,copy=0.;
        for(int k=tl;k<th;k++) {mean+=prob[k-tl]*values[k];copy+=prob[k-tl]*zero_delay[k];}
        double *out=trace+8*sweep;out[0]=mean;out[1]=copy;out[2]=values[old];
        for(int a=0;a<3;a++) out[3+a]=total[a];out[6]=count;
        double q=0.;
        for(int a=0;a<3;a++) for(int b=0;b<3;b++) q+=(observed[a]-total[a])*inverse[3*a+b]*(observed[b]-total[b]);
        out[7]=-.5*(5+dims)*log1p(fmax(q,0.)/5.);
    }
    free(prob);return 0;
}
