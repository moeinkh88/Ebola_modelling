# this code is for finding initial guess for the acceptable interval of α2 and μ2

using Plots,SpecialFunctions, StatsBase, Random
using DifferentialEquations, Turing, LinearAlgebra
# Load StatsPlots for visualizations and diagnostics.
using StatsPlots
#initial conditons and parameters
using CSV
using DataFrames
using FdeSolver, Plots,SpecialFunctions, Optim, StatsBase, Random
using Interpolations
# Dataset
Data=CSV.read("datoswho.csv", DataFrame)

data=(Matrix(Float64.(Data)))
#initial conditons and parameters


x0=[18000,0,15,0,0,0,0,0]# initial conditons S0,E0,I0,R0,L0,H0,B0,C0
N=sum(x0)

α1=3.537e-2 # Density independent part of the birth rate for individuals.
α2=.1*α1 # Density dependent part of the birth rate for individuals.
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
βr=0.305 # Contact rate of infective individuals and asymptomatic.
# βr=0.185
μ=14e-3
μ1=10.17e-3 # Density independent part of the death rate for individuals.
μ2=.1*μ1 # Density dependent part of the death rate for individuals.
ξ=14e-3 # Incineration rate

par1=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,N] # for model with constant N
par2=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,α2,μ2] # for model with variable N
# par3=[α1,σ,γ1,γ2,ϵ,δ1,δ2,τ,βi,βd,βh,μ1,ξ] # I need it for fitting
par3=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,μ1,ξ] # I need it for fitting

tSpan=(4,438)

#Define the equation

function  F1(dx, x, par, t) # model with constant N

    α1,σ,γ1,γ2,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,N=par
	S, E, I, R, L, H, B, C=x

    dx[1]=μ*N - βi/N*S*I - βh/N*S*H - βd/N*S*L - βr/N*S*R - μ*S
    dx[2]=βi/N*S*I + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - μ*E
    dx[3]=σ*E - (γ1 + ϵ + τ + μ)*I
    dx[4]=γ1*I + γ2*H - (γ3 + μ)*R
    dx[5]=ϵ*I - (δ1+ξ)*L
    dx[6]=τ*I - (γ2 + δ2 + μ)*H
    dx[7]=δ1*L + δ2*H - ξ*B
    dx[8]=γ3*R - μ*C

    return nothing

end

function  F2(dx, x, par, t) # model with variable N

    α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,μ1,ξ,βr,α2,μ2=par
    S, E, I, R, L, H, B, C=x
    N=sum(x)

    dx[1]=(α1-α2*N)*N - βi/N*S*I - βh/N*S*H - βd/N*S*L - βr/N*S*R - (μ1+μ2*N)*S
    dx[2]=βi/N*S*I + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - (μ1+μ2*N)*E
    dx[3]=σ*E - (γ1 + ϵ + τ + μ1 + μ2*N)*I
    dx[4]=γ1*I + γ2*H - (γ3 + μ1 + μ2*N)*R
    dx[5]=ϵ*I - (δ1+ξ)*L
    dx[6]=τ*I - (γ2 + δ2 + μ1 + μ2*N)*H
    dx[7]=δ1*L + δ2*H - ξ*B
    dx[8]=γ3*R - (μ1 + μ2*N)*C

    return nothing

end

## optimazation of μ2 and α2 for integer order model
prob1 = ODEProblem(F1, x0, tSpan, par1)
prob2 = ODEProblem(F2, x0, tSpan, par2)

sol = solve(prob1, alg_hints=[:stiff]; saveat=0.1)
RealData=LinearInterpolation(data[:,1], data[:,2])


@model function fitlv(data, prob2)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α2 ~ truncated(Normal(0, .0001); lower=0, upper=.0001)
    μ2 ~ truncated(Normal(0, .0001); lower=0, upper=.0001)
	βr ~ truncated(Normal(0.1, .9); lower=0.1, upper=.9)
	# γ3 ~ truncated(Normal(0.001, .9); lower=0.001, upper=.9)

    # Simulate model.
	# p=vcat(par3,γ3,βr,α2,μ2)
	p=vcat(par3,βr,α2,μ2)
    xx = solve(prob2, alg_hints=[:stiff],reltol=1e-8,abstol=1e-8; p=p, saveat=1)
	x=hcat(xx.u...)
	S=x[1,:]; E=x[2,:]; I1=x[3,:]; R=x[4,:];
	L=x[5,:]; H=x[6,:]; B=x[7,:]; C=x[8,:];
	N=sum(x,dims=1)'
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx=@.I1+R+L+H+B+C-μ3[:,1]*(N[:,1]-S-E)
    # Observations.
    for i in 1:length(S)
        data[i] ~ Normal(Appx[i], σ^2)
    end

    return nothing
end

model = fitlv(RealData(4:438), prob2)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(0.65), MCMCSerial(), 2000, 3; progress=false)


#plotting
pltChain4=plot(chain)
savefig(pltChain4,"pltChain4.svg")

# Save a chain.
write("chain-file2.jls", chain)

# Read a chain.
chn2 = read("chain-file2.jls", Chains)

plot(; legend=false)
posterior_samples = sample(chain[[:α2, :μ2]], 2000; replace=false)
for p1 in eachrow(Array(posterior_samples))
	p=vcat(par3,p1[1],p1[2])
    sol_p = solve(prob2, Tsit5(); p=p, saveat=0.1)
	x=reduce(vcat,sol_p.u')
	S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N1=sum(x,dims=2)
	μ3=mean([μ1 .+ p1[2] .* N1, α1 .- p1[1] .*N1])
	Appx=@.I1+R+L+H+B+C-μ3*(N1-S-E)
    plot!(sol_p.t,Appx; alpha=0.1, color="#BBBBBB")
end
# Plot simulation and noisy observations.
x=reduce(vcat,sol.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
Ex=@.I1+R+L+H+B+C-μ*(N-S-E)
plt305=scatter!(data[:,1],data[:,2]; color=[1 2], linewidth=1)

savefig(plt305,"pltFit305.svg")

#optimized values

# mean(chain[:α2])
α2=8.334570273681649e-8

# mean(chain[:μ2])
μ2=4.859965522407347e-7

# mean(chain[:βr])
βr=0.2174851498937417

#let's plot the results of two models
x=reduce(vcat,sol.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
Ex=@.I1+R+L+H+B+C-μ*(N-S-E)
plot(sol.t, Ex; lw=1, label="Model1")

# p=vcat(par3,γ3,βr,α2,μ2)
p=vcat(par3,βr,α2,μ2)
# p=vcat(par3,p1[1],p1[2])
sol_p = solve(prob2, alg_hints=[:stiff]; p=p, saveat=0.1)
x=reduce(vcat,sol_p.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx=@.I1+R+L+H+B+C-μ3*(N1-S-E)
plot!(sol_p.t,Appx, label="Model2")
scatter!(data[:,1],data[:,2],legendposition=:right)

plot(sol)
plot(N1)

using CSV
using DataFrames
using Interpolations
# Dataset
Data=CSV.read("plot-data.csv", DataFrame)

data=(Matrix(Data))
Data2=LinearInterpolation(data[:,1], data[:,2])
plt=scatter!(Data2(1:85),label="Real data",xlabel="days", ylabel="Cumulative confirmed cases",
		legendposition=:bottomright)

savefig(plt,"plt.png")
pltN=plot(sol_p.t,N1,ylabel="Population N",xlabel="days",
			legend=:false)

savefig(pltN,"pltN.png")
#################for βr=305

function  F11(dx, x, par, t) # model with constant N

    α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,N=par
	# βr=0.185 # why is it different value in the papers
    S, E, I, R, L, H, B, C=x

    dx[1]=μ*N - βi/N*S*I - βh/N*S*H - βd/N*S*L - βr/N*S*R - μ*S
    dx[2]=βi/N*S*I + βh/N*S*H + βd/N*S*L + βr/N*S*R - σ*E - μ*E
    dx[3]=σ*E - (γ1 + ϵ + τ + μ)*I
    dx[4]=γ1*I + γ2*H - (γ3 + μ)*R
    dx[5]=ϵ*I - (δ1+ξ)*L
    dx[6]=τ*I - (γ2 + δ2 + μ)*H
    dx[7]=δ1*L + δ2*H - ξ*B
    dx[8]=γ3*R - μ*C

    return nothing

end
prob1 = ODEProblem(F11, x0, tSpan, par1)

sol = solve(prob1, alg_hints=[:stiff]; saveat=0.1)
x=reduce(vcat,sol.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
Ex=@.I1+R+L+H+B+C-μ*(N-S-E)
plot(sol.t, Ex; label="Model1", lw=3)

p1=[1.1862350479039923e-6,2.1259887426666564e-7]
p=vcat(par3,p1[1],p1[2])
sol_p = solve(prob2, alg_hints=[:stiff]; p=p, saveat=0.1)
x=reduce(vcat,sol_p.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ p1[2] .* N1, α1 .- p1[1] .*N1])
Appx=@.I1+R+L+H+B+C-μ3*(N1-S-E)
plot!(sol_p.t,Appx, label="Model2",lw=3, linestyle=:dash)


using CSV
using DataFrames
using Interpolations
# Dataset
Data=CSV.read("plot-data.csv", DataFrame)

data=(Matrix(Data))
Data2=LinearInterpolation(data[:,1], data[:,2])
plt=scatter!(Data2(1:85),label="Real data",xlabel="days", ylabel="Cumulative confirmed cases",
		legendposition=:left)

savefig(plt,"plt2.png")
pltN2=plot(sol_p.t,N1,ylabel="Population N",xlabel="days",
			legend=:false)

savefig(pltN2,"pltN2.png")
