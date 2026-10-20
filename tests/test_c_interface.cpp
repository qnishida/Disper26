#include "../src/disper26_c.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool ok) { if (!ok) throw std::runtime_error("C interface assertion failed"); }
int main() {
    const double p[]={2.5,67.6,22.6,67.6,22.5,22.5},h[]={10};
    const double hs[]={3.3,211.2,77.55,211.2,66.825,66.825};
    double roots[20],value;
    int count=0;
    char message[256];
    int status=disper26_search_roots(1,p,h,hs,0.3141592653589793,1,4.49,500,0,
                                    nullptr,0,&count,message,256);
    check(status==DISPER26_BUFFER_TOO_SMALL && count>0 && message[0]);
    status=disper26_search_roots(1,p,h,hs,0.3141592653589793,1,4.49,500,0,
                               roots,20,&count,message,256);
    check(status==DISPER26_OK && count>0 && message[0]=='\0');
    char small[2]={'x','x'};
    status=disper26_compute_hv_ratio(1,p,h,hs,1,3,99,&value,small,2);
    check(status==DISPER26_INVALID_ARGUMENT && std::isnan(value) && small[1]=='\0');
    status=disper26_compute_group_velocity_ad(1,nullptr,h,hs,1,3,0,&value,message,256);
    check(status==DISPER26_INVALID_ARGUMENT && std::isnan(value));
    status=disper26_compute_hv_ratio(1,p,h,hs,1,3,1,&value,nullptr,0);
    check(status==DISPER26_OK && std::isnan(value));
    status=disper26_search_roots(1,p,h,hs,1,1,4.49,500,0,roots,20,nullptr,message,256);
    check(status==DISPER26_INVALID_ARGUMENT);
    const double z[]={0,10}, fields[]={2.5,5.2,3,3.3,8,4.5};
    const int flags[]={0,0};
    double c,u,hv,error;
    int level;
    status=disper26_compute_dispersion_romberg_velocity_iso(2,z,fields,flags,20,0,3,5,1e-10,0,1,0,0,
        nullptr,&u,&hv,&level,&error,message,256);
    check(status==DISPER26_INVALID_ARGUMENT && std::isnan(u) && level==-1);
    status=disper26_compute_dispersion_romberg_velocity_iso(2,z,fields,flags,20,0,3,-1,1e-10,0,1,0,0,
        &c,&u,&hv,&level,&error,message,256);
    check(status==DISPER26_INVALID_ARGUMENT && std::isnan(c));
    std::cout << "C buffer and error checks passed\n";
}
