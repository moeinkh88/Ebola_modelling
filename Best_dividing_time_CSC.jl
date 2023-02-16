#for finding optimized orders

using CSV
using DataFrames
using FdeSolver, SpecialFunctions, Optim, StatsBase, Random
using Interpolations

# Dataset
Data=CSV.read("datoswho.csv", DataFrame)

data=(Matrix(Float64.(Data)))
#initial conditons and parameters

x0=[18000,0,15,0,0,0,0,0]# initial conditons S0,E0,I0,R0,L0,H0,B0,C0

α1=3.537e-2 # Density independent part of the birth rate for individuals.
# α2=2.6297211237376283e-7# Density dependent part of the birth rate for individuals.
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
# βr=0.185 # Contact rate of infective individuals and asymptomatic.
μ=14e-3
μ1=10.17e-3 # Density independent part of the death rate for individuals.
# μ2=3.750689119502623e-7 # Density dependent part of the death rate for individuals.
ξ=14e-3 # Incineration rate


α2=8.334570273681649e-8
μ2=4.859965522407347e-7
βr=0.2174851498937417

par=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,α2,μ2]

i=parse(Int32,ARGS[1])
T=216:300
tSpan=[4,T[i]]
tSpan2=[T[i],438]

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

##
Data2=LinearInterpolation(data[:,1], data[:,2])

t1, x1 = FDEsolver(F, tSpan, x0, ones(8), par,h=.01,nc=4)
x02=x1[end,:]


function loss_2(p)# loss function

	α=p
	if size(x0,2) != Int64(ceil(maximum(α))) # to prevent any errors regarding orders higher than 1
	indx=findall(x-> x>1, α)
	α[indx]=ones(length(indx))
	end
	_, x = FDEsolver(F, tSpan2, x02, α, par,h=.01,nc=4)
	S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx=@.I1+R+L+H+B+C-μ3*(N-S-E)
    rmsd(Appx[1:10:end,:], Data2(T[i]:.1:438); normalize=:true) # Normalized root-mean-square error
end

p_lo_1=[.5, .5, .5, .5, .5, .5, .5, .5] #lower bound
p_up_1=[1, 1, 1, 1, 1, 1, 1, 1] # upper bound
p_vec_1=[0.939531849631367, 0.9998860578731849, 0.9997741401361016, 0.9999076787478497, 0.991141852453178, 0.9990686884413122, 0.9432066029272619, 0.7936761619660204]
Res2=optimize(loss_2,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS())# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res2=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.99), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			# Optim.Options(outer_iterations = 10,
			# 			  iterations=3,
			# 			  show_trace=true,
			# 			  show_every=1)
			)
# display(Res2)
α=vcat(Optim.minimizer(Res2))
# display(α)
#error estimation
t2, x2= FDEsolver(F, [T[i],438], x02, α, par, nc=4,h=.01)

x=vcat(x1,x2[2:end,:])
t=vcat(t1,t2[2:end,:])
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
Appx=@.I1+R+L+H+B+C-μ3*(N-S-E)
Err=rmsd(Appx[1:10:end,:], Data2(4:.1:438); normalize=:true) # Normalized root-mean-square error

display([Err i])
