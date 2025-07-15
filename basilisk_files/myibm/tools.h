/**
Implementation in basilisk from:
Lee, W.; Lee, S. Immersed Boundary Method for Simulating Interfacial Problems. Mathematics 2020, 8, 1982. https://doi.org/10.3390/math8111982
*/

#include "common.h"
#include "fractions.h"


double phi_func_Roma(double r){                                              // From Roma, 1999 : https://www.sciencedirect.com/science/article/pii/S0021999199962939
    double val;
    if (fabs(r)<0.5){
        val = 1./3.*(1. + sqrt(-3 * sq(r) + 1));
    }else if (fabs(r)<1.5) {
        val = 1./6.*(5. - 3.*fabs(r)+sqrt(-3*sq(1 - fabs(r)) + 1));
    }else{
        val = 0.;
    }
    return val;
}


double phi_func_Peskin(double r){
    double val;
    if (fabs(r)<1){
        val = 1./8.*(3. - 2.*fabs(r)+sqrt(1.+4.*fabs(r)-4.*sq(r)));
    }else if (fabs(r)<2) {
        val = 1./8.*(5. - 2.*fabs(r)-sqrt(-7.+12.*fabs(r)-4.*sq(r)));
    }else{
        val = 0.;
    }
    return val;
}


double dirach_delta_func(double dx, coord pos1, coord pos2){
    if (dx<=0){
        fprintf(stderr, "Error: dx must be larger than 0 (Grid resolution)");
        exit(EXIT_FAILURE);
    }
    double val = 0;
    foreach_dimension(){
        val *= 1/dx*phi_func_Roma((pos1.x-pos2.x)/dx);
    }
    return val;
}


void divergence(vector v, scalar div){
    foreach(){
        div[] = 0.;
        foreach_dimension(){
        div[] += (v.x[1] - v.x[-1])/(2*Delta);
        }
    }
}

void gradient(scalar s, vector grad_s){
    foreach(){
        foreach_dimension(){
            grad_s.x[] = (s[1] - s[-1]) / (2*Delta);
        }
    }
}

void scalar_laplacian(scalar s, scalar lap){
    foreach(){
        #if dimension == 2
        lap[] = (s[-1,0] + s[1,0] + s[0,-1] + s[0,1] - 4*s[])/sq(Delta);
        #elif dimension == 3
        lap[] = (s[-1,0,0] + s[1,0,0] + s[0,-1,0] + s[0,1,0] + s[0,0,1] + s[0,0,-1] - 6*s[])/sq(Delta);
        #endif
    }
}
