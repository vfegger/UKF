#include "../include/HeatFluxEstimationMemory.hpp"

HeatFluxEstimationMemory::HeatFluxEstimationMemory() : UKFMemory() {

}

HeatFluxEstimationMemory::HeatFluxEstimationMemory(Data& a, DataCovariance&b, DataCovariance&c, Data&d, DataCovariance&e, Parameter&f, PointerType type_in, PointerContext context_in) : UKFMemory(a,b,c,d,e,f,type_in,context_in){

}

inline double HeatFluxEstimationMemory::C(double T){
        return 1324.75*T+3557900.0;
}
inline double HeatFluxEstimationMemory::K(double T){
        return 12.45 + (14e-3 + 2.517e-6*T)*T;
}
double HeatFluxEstimationMemory::DifferentialK(double TN, double T, double TP, double delta){
    double auxN = 2.0*(K(TN)*K(T))/(K(TN)+K(T))*(TN - T)/delta;
    double auxP = 2.0*(K(TP)*K(T))/(K(TP)+K(T))*(TP - T)/delta;
    return (auxN+auxP)/delta;
}
void HeatFluxEstimationMemory::Evolution(Data& data_inout, Parameter& parameter_in) {
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    unsigned Lx = length.pointer[0u];
    unsigned Ly = length.pointer[1u];
    unsigned Lz = length.pointer[2u];
    unsigned Lt = length.pointer[3u];
    Pointer<double> delta = parameter_in.GetPointer<double>(1u);
    double dx = delta.pointer[0u];
    double dy = delta.pointer[1u];
    double dz = delta.pointer[2u];
    double dt = delta.pointer[3u];
    double amp = parameter_in.GetPointer<double>(3u).pointer[0u];
    double acc = 0.0;
    double TiN, TiP;
    double TjN, TjP;
    double TkN, TkP;
    double T0;
    unsigned index;
    Pointer<double> T = data_inout[0u];
    Pointer<double> Q = data_inout[1u];
    Pointer<double> T_out = MemoryHandler::Alloc<double>(data_inout.GetLength(0u),T.type,T.context);
    for(unsigned k = 0u; k < Lz; k++){
        for(unsigned j = 0u; j < Ly; j++){
            for(unsigned i = 0u; i < Lx; i++){
                index = (k*Ly+j)*Lx+i;
                T0 = T.pointer[index];
                TiN = (i != 0u  ) ? T.pointer[index - 1]     : T0;
                TiP = (i != Lx-1) ? T.pointer[index + 1]     : T0;
                TjN = (j != 0u  ) ? T.pointer[index - Lx]    : T0;
                TjP = (j != Ly-1) ? T.pointer[index + Lx]    : T0;
                TkN = (k != 0u  ) ? T.pointer[index - Ly*Lx] : T0;
                TkP = (k != Lz-1) ? T.pointer[index + Ly*Lx] : T0;
                acc = 0.0;
                // X dependency
                acc += DifferentialK(TiN,T0,TiP,dx);
                // Y dependency
                acc += DifferentialK(TjN,T0,TjP,dy);
                // Z dependency
                acc += DifferentialK(TkN,T0,TkP,dz);
                if(k == Lz - 1){
                    acc += amp*Q.pointer[j*Lx+i]/dz;
                }
                T_out.pointer[index] = T0 + dt*acc/C(T0);
            }
        }
    }
    for(unsigned k = 0u; k < Lz; k++){
        for(unsigned j = 0u; j < Ly; j++){
            for(unsigned i = 0u; i < Lx; i++){
                index = (k*Ly+j)*Lx+i;
                T.pointer[index] = T_out.pointer[index];
            }
        }
    }
    MemoryHandler::Free<double>(T_out);
}

void HeatFluxEstimationMemory::Observation(Data& data_in, Parameter& parameter_in, Data& data_out) {
    Pointer<unsigned> length = parameter_in.GetPointer<unsigned>(0u);
    unsigned Lx = length.pointer[0u];
    unsigned Ly = length.pointer[1u];
    Pointer<double> T_in = data_in[0u];
    Pointer<double> T_out = data_out[0u];
    unsigned index;
    for(unsigned j = 0u; j < Ly; j++){
        for(unsigned i = 0u; i < Lx; i++){
            index = j*Lx+i;
            T_out.pointer[index] = T_in.pointer[index];
        }
    }
}