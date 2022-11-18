using FdeSolver, Plots, Statistics
using CSV
using DataFrames,StatsBase
using Interpolations,LinearAlgebra

#initial conditons and parameters

x0=[18000,0,15,0,0,0,0,0]# initial conditons S0,E0,I0,R0,L0,H0,B0,C0

α1=35.37e-3 # Density independent part of the birth rate for individuals.
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
μ1=10.17e-3 # Density independent part of the death rate for individuals.
ξ=14e-3 # Incineration rate
μ=14e-3
#fitted 3 parameters by Turing (HMC)
α2=8.334570273681649e-8
μ2=4.859965522407347e-7
βr=0.2174851498937417

par=[α1,α2,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,μ2,ξ]
#fitted order based on 3 optimized parameters
# α=[0.9414876125410915, 0.999999999635505, 0.999999999265876, 0.9999999997019355, 0.9999999682296614, 0.999999996978037, 0.9430213826071122, 0.793233020972691]
α=[0.9414876354377308, 0.9999999999997804, 0.999999999999543, 0.9999999999998147, 0.9999999999806818, 0.9999999999981127, 0.9430213976522912, 0.7932329945305198]
T=438
tSpan=[4,T]

#Define the equation

function  F(t, x, par)

    α1,α2,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,μ2,ξ=par
    S, E, I, R, L, H, B, C=x
    N=sum(x)

    dS=(α1-α2*N)*N - βi/N*S*I - βh/N*S*H - βd/N*S*L - βr/N*S*R - (μ1+μ2*N)*S
    dE=βi/N*S*I + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - (μ1+μ2*N)*E
    dI=σ*E - (γ1 + ϵ + τ + μ1 + μ2*N)*I
    dR=γ1*I + γ2*H - (γ3 + μ1 + μ2*N)*R
    dL=ϵ*I - (δ1+ξ)*L
    dH=τ*I - (γ2 + δ2 + μ1 + μ2*N)*H
    dB=δ1*L + δ2*H - ξ*B
    dC=γ3*R - (μ1 + μ2*N)*C

    return [dS, dE, dI, dR, dL, dH, dB, dC]

end
#eq with constant N
parC=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,sum(x0)]
function  Fc(t, x, par)

    α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,N=par
    S, E, I, R, L, H, B, C=x
	βr=0.185
	γ3=1/30
	μ=14e-3

    dS=μ*N - βi/N*S*I - βh/N*S*H - βd/N*S*L - βr/N*S*R - μ*S
    dE=βi/N*S*I + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - μ*E
    dI=σ*E - (γ1 + ϵ + τ + μ)*I
    dR=γ1*I + γ2*H - (γ3 + μ)*R
    dL=ϵ*I - (δ1+ξ)*L
    dH=τ*I - (γ2 + δ2 + μ)*H
    dB=δ1*L + δ2*H - ξ*B
    dC=γ3*R - μ*C

    return [dS, dE, dI, dR, dL, dH, dB, dC]

end
#solving

#solve Eq constant N
_, x = FDEsolver(Fc, tSpan, x0, ones(8), parC,h=.01,nc=4)
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N1C=sum(x,dims=2)
AppxC=@.I1+R+L+H+B+C-μ*(N1C-S-E)


#model with optimized orders
t, x= FDEsolver(F, tSpan, x0, α, par, nc=4,h=.01)
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
app=@.I1+R+L+H+B+C-μ3*(N-S-E)

#lets test with integer order
_, x = FDEsolver(F, tSpan, x0, ones(8), par,h=.01,nc=4)
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx1=@.I1+R+L+H+B+C-μ3*(N1-S-E)

# Dataset
Data=CSV.read("datoswho.csv", DataFrame)
data=(Matrix(Float64.(Data)))
Data2=LinearInterpolation(data[:,1], data[:,2])


Err=rmsd(app[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
Err1=rmsd(Appx1[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
Err0=rmsd(AppxC[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
#plotting
plt=plot(t,x,xlabel="time",lw = 2, label=["S" "E"  "I"  "R"  "L" "H" "B" "C"],
    thickness_scaling =1.2)
savefig(plt,"plt.png")

scatter(data[:,1],data[:,2],label="Real data",c="black")
plot!(t,Appx1, lw=3,label="Model 1", xlabel="Time (days)", ylabel="Cumulative confirmed cases")
plot!(t,app,lw=3,label="Model f8",linestyle=:dash,xlabel="time",legendposition=:bottomright)
pltData=plot!(t,AppxC, lw=3,label="Model 0")
savefig(pltData,"pltData.png")

#plot population
plot(t,N,lw=3,label="Model 1",xlabel="Days",ylabel="Population")
pltN=plot!(t,N1,lw=3,linestyle=:dash,label="Model f8",legendposition=:right)
savefig(pltN,"pltN.png")

# Norm plot
Mf8=(app[1:10:end,:] - Data2(4:.1:T))
M1=(Appx1[1:10:end,:] - Data2(4:.1:T))

plot(t[1:10:end,:],M1,label="Model 1", xlabel="Days",ylabel="Error (appX - exactX)")
pltErr=plot!(t[1:10:end,:],Mf8,label="Model f8")
savefig(pltErr,"pltErr.png")

norm(Mf8,2)
norm(M1,2)
