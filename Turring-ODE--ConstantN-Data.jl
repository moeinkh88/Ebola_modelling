# this code is for for optimizing μ and βr

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
μ=14e-3
ξ=14e-3 # Incineration rate

par1=[σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,ξ,N,βr,μ] # for model with constant N
par=[σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,ξ,N] # for fitting

tSpan=(4,438)

#Define the equation

function  F1(dx, x, par, t) # model with constant N

    σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,ξ,N,βr,μ=par
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

## optimazation of μ2 and α2 for integer order model
prob1 = ODEProblem(F1, x0, tSpan, par1)

sol = solve(prob1, alg_hints=[:stiff]; saveat=0.1)
RealData=LinearInterpolation(data[:,1], data[:,2])


@model function fitlv(data, prob2)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    βr ~ truncated(Normal(0.1, .9); lower=0.1, upper=.9)
	μ ~ truncated(Normal(0, .1); lower=0, upper=.1)

    # Simulate model.
	p=vcat(par,βr,μ)
    xx = solve(prob2, alg_hints=[:stiff],reltol=1e-8,abstol=1e-8; p=p, saveat=1)
	x=hcat(xx.u...)
	S=x[1,:]; E=x[2,:]; I1=x[3,:]; R=x[4,:];
	L=x[5,:]; H=x[6,:]; B=x[7,:]; C=x[8,:];
	Appx=@.I1+R+L+H+B+C-μ*(N-S-E)
    # Observations.
    for i in 1:length(S)
        data[i] ~ Normal(Appx[i], σ^2)
    end

    return nothing
end

model = fitlv(RealData(4:438), prob1)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(0.65), MCMCSerial(), 2000, 3; progress=false)


#plotting
pltChainConstN=plot(chain)
savefig(pltChainConstN,"pltChainConstN.svg")

# # Save a chain.
# write("chain-file2.jls", chain)
#
# # Read a chain.
# chn2 = read("chain-file2.jls", Chains)
# julia> mean(chain[:βr])
βr=0.7265002737432911
# julia> mean(chain[:μ])
μ=0.06918229886616623

plot(; legend=false)
posterior_samples = sample(chain[[:βr, :μ]], 2000; replace=false)
for p1 in eachrow(Array(posterior_samples))
	p=vcat(par,p1[1],p1[2])
    sol_p = solve(prob1, Tsit5(); p=p, saveat=0.1)
	x=reduce(vcat,sol_p.u')
	S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	Appx=@.I1+R+L+H+B+C-μ*(N-S-E)
    plot!(sol_p.t,Appx; alpha=0.1, color="#BBBBBB")
end
# Plot simulation and noisy observations.
x=reduce(vcat,sol.u')
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
Ex=@.I1+R+L+H+B+C-μ*(N-S-E)
pltConstN=scatter!(data[:,1],data[:,2]; color=[1 2], linewidth=1)

savefig(pltConstN,"pltConstN.svg")


# \optimzed values'
βr=0.7265002737432911
μ=0.06918229886616623
