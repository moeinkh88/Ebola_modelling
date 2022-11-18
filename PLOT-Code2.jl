using CSV
using DataFrames
using FdeSolver, Plots,SpecialFunctions, Optim, StatsBase, Random
using Interpolations, LinearAlgebra
# Dataset
Data=CSV.read("datoswho.csv", DataFrame)
data=(Matrix(Float64.(Data)))
Data2=LinearInterpolation(data[:,1], data[:,2]) #Interpolate the data
#initial conditons and parameters

x0=[18000,0,15,0,0,0,0,0]# initial conditons S0,E0,I0,R0,L0,H0,B0,C0

α1=35.37e-3 # Density independent part of the birth rate for individuals.
α2=8.334570273681649e-8# Density dependent part of the birth rate for individuals.
σ=1/11.4 # Per capita rate at which exposed individuals become infectious.
γ1=0.1 # Per capita rate of progression of individuals from the infectious class to the asymptomatic class.
γ2=1/5 # Per capita rate of progression of individuals from the hospitalized class to the asymptomatic class.
γ3=1/30 # Per capita recovery rate of individuals from the asymptomatic class to the complete recovered class.
ϵ=1/9.6 # Fatality rate.
δ1=1/2 # Per capita rate of progression of individuals from the dead class to the buried class.
δ2=1/4.6 # Per capita rate of progression of individuals from the hospitalized class to the buried class.
τ=1/5 # Per capita rate of progression of individuals from the infectious class to the hospitalized class.
βi=0.14 # Contact rate of infective individuals and susceptible.
βd=0.4 # Contact rate of infective individuals and dead.
βh=0.29 # Contact rate of infective individuals and hospitalized.
βr=0.2174851498937417 # Contact rate of infective individuals and asymptomatic.
μ=14e-3
μ1=10.17e-3 # Density independent part of the death rate for individuals.
μ2=4.859965522407347e-7 # Density dependent part of the death rate for individuals.
ξ=14e-3 # Incineration rate

par=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,α2,μ2]

#Define the equation
function  F(t, x, par)

    α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,α2,μ2=par
    S, E, I1, R, L, H, B, C=x
    N=sum(x)

    dS=(α1-α2*N)*N - βi/N*S*I1 - βh/N*S*H - βd/N*S*L - βr/N*S*R - (μ1+μ2*N)*S
    dE=βi/N*S*I1 + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - (μ1+μ2*N)*E
    dI=σ*E - (γ1 + ϵ + τ + μ1 + μ2*N)*I1
    dR=γ1*I1 + γ2*H - (γ3 + μ1 + μ2*N)*R
    dL=ϵ*I1 - (δ1+ξ)*L
    dH=τ*I1 - (γ2 + δ2 + μ1 + μ2*N)*H
    dB=δ1*L + δ2*H - ξ*B
    dC=γ3*R - (μ1 + μ2*N)*C

    return [dS, dE, dI, dR, dL, dH, dB, dC]

end

#optimized order of derivatives
# α=[0.9999826367080396, 0.583200182055744, 0.9999997090382831, 0.9999552779111426, 0.9999931233485699, 0.999931061595331, 0.9999999960137106, 0.9999997372162407]
α=ones(8);α[2]=0.583200182055744
T1=250
T2=438
tSpan1=[4,T1]
tSpan2=[T1,T2]

#ODE
t, x= FDEsolver(F, [4,T2], x0, ones(8), par, nc=4,h=.01)
S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
Err1=rmsd(Appx1[1:10:end,:], Data2(4:.1:T2)) # Normalized root-mean-square error

# 2 sections: first integer until 250 then fractional
t1, x1= FDEsolver(F, tSpan1, x0, ones(8), par, nc=4,h=.01)
x02=x1[end,:]
t2, x2= FDEsolver(F, tSpan2, x02, α, par, nc=4,h=.01)
xf=vcat(x1,x2[2:end,:])
t=vcat(t1,t2[2:end,:])
S=xf[:,1]; E=xf[:,2]; I=xf[:,3]; R=xf[:,4];
L=xf[:,5]; H=xf[:,6]; B=xf[:,7]; C=xf[:,8];
N=sum(xf,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
Appx=@.I+R+L+H+B+C-μ3*(N-S-E)
Err=rmsd(Appx[1:10:end,:], Data2(4:.1:T2)) # Normalized root-mean-square error

# Norm/abs
# Mf8=@.abs(Appx[1:10:end,:] - Data2(4:.1:T2))
# M1=@.abs(Appx1[1:10:end,:] - Data2(4:.1:T2))


#plotting
t, x= FDEsolver(F, [4,450], x0, ones(8), par, nc=4,h=.01)
S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)

t1, x1= FDEsolver(F, tSpan1, x0, ones(8), par, nc=4,h=.01)
x02=x1[end,:]
t2, x2= FDEsolver(F, [T1,450], x02, α, par, nc=4,h=.01)
xf=vcat(x1,x2[2:end,:])
t=vcat(t1,t2[2:end,:])
S=xf[:,1]; E=xf[:,2]; I=xf[:,3]; R=xf[:,4];
L=xf[:,5]; H=xf[:,6]; B=xf[:,7]; C=xf[:,8];
N=sum(xf,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
Appx=@.I+R+L+H+B+C-μ3*(N-S-E)

scatter(data[:,1],data[:,2], label="Real data", c="khaki3",markerstrokewidth=0)
plot!(t,Appx1,ylabel="Cumulative confirmed cases",lw=2, label="Integer model, RMSD=276.5", c="darkorange3")
plot!(t,Appx,xlabel="Days", legendposition=:bottomright, lw=2, label="Fractional model, RMSD=211.0"
    , linestyle=:dot, c="blue1")

#plot population
plot(t,N1,c="darkorange3",lw=4,legendposition=:right,label="Integer model")
plot!(t,N, c="blue1", label="Fractional model",linestyle=:dot, lw=4, xlabel="Days", ylabel="Variable population")

#
# plot(t[1:10:end,:],Mf8)
# plot!(t[1:10:end,:],M1)
#
