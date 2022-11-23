# EBOLA VIRUS DISEASE DYNAMICS WITH SOME
# PREVENTIVE MEASURES: A CASE STUDY
# OF THE 2018–2020 KIVU OUTBREAK

using FdeSolver, Plots, Statistics
using CSV
using DataFrames
using Interpolations,LinearAlgebra

#initial conditons and parameters

x0=[11000000,300,300,300,400,200,5] # initial conditons S0,J0,I0,H0,D0,R0,V0

PI=400
p=0.5708
τ=0.005
θ=0.9
β=0.16
ν1=1.5267
ν2=1.5
ϵ=0.3875
γj=0.3796
σ1=0.3074
σ2=0.7086
ηj=0.2311
μ=10.13/1000
ηi=0.25
δi=0.3079
δj=0.0152
δh=0.1808
γh=0.4594
γi=0.0153
α=0.5723
b=1/2.01
u=0.4019
q1=0.05
q2=0.05
q3=0.1173
q4=0.06
βv=5.07e-8

σh = ϵ*(1 - σ1)
θ1 = θ*τ
σd = ν2*(1 - σ2)
θ2 = μ + θ*τ
φ1 = (α + δj + γj + ηj)
φ3 = (γh + δh)
φ2 = (δi + ηi)

par=[PI,p,τ,θ,β,ν1,ν2,ϵ,γj,σ1,σ2,ηj,μ,ηi,δi,δj,δh,γh,γi,α,b,u,q1,q2,q3,q4,βv]

tSpan=[0,50]

#Define the equation

function  F(t, x, par)

    PI,p,τ,θ,β,ν1,ν2,ϵ,γj,σ1,σ2,ηj,μ,ηi,δi,δj,δh,γh,γi,α,b,u,q1,q2,q3,q4,βv=par
    S, J, I, H, D, R, V=copy(x)
    N=sum(x[1:6])

    λ=(β*(J + ϵ*(1-σ1)*H + ν1*I + ν2*(1-σ2)*D))/N+βv*V

    dS = PI - λ*S - θ2*S
    dJ= p*λ*S - φ1*J
    dI=(1-p)*λ*S + α*J - φ2*I
    dH= ηj*J + ηi*I - φ3*H
    dD=δj*J + δi*I + δh*H - b*D
    dR=θ1*S + γj*J + γi*I + γh*H - μ*R
    dV=q1*J + q2*I + q3*H + q4*D - u*V

    return [dS, dJ, dI, dH, dD, dR, dV]

end

#solving
t, x= FDEsolver(F, tSpan, x0, ones(7), par, nc=4,h=.01)
plot(t,sum(x[:,2:6],dims=2))
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4]
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8]
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

#plotting
plt=plot(t,x,xlabel="time",lw = 2, label=["S" "E"  "I"  "R"  "L" "H" "B" "C"],
    thickness_scaling =1.2)
# savefig(plt,"plt.png")

Err=rmsd(app[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
Err1=rmsd(Appx1[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
scatter(data[:,1],data[:,2],label="realdata",c="black")
plot!(t,app, lw=3,label="model1")
plot!(t,Appx1,lw=3,label="model2",linestyle=:dash,xlabel="time",legendposition=:right)
#plot population
plot(t,N,lw=3)
plot!(t,N1,lw=3,linestyle=:dash)


# Norm plot
Mf8=@.abs(Appx[1:10:end,:] - Data2(4:.1:T))
M1=@.abs(Appx1[1:10:end,:] - Data2(4:.1:T))

plot(t[1:10:end,:],Mf8)
plot!(t[1:10:end,:],M1)
