#for finding optimized orders

using CSV
using DataFrames
using FdeSolver, Plots,SpecialFunctions, Optim, StatsBase, Random
using Interpolations
# Dataset
Data=CSV.read("datoswho.csv", DataFrame)

data=(Matrix(Float64.(Data)))
#initial conditons and parameters

x0=[18000,0,15,0,0,0,0,0]# initial conditons S0,E0,I0,R0,L0,H0,B0,C0

α1=35.37e-3 # Density independent part of the birth rate for individuals.
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
# Par=vcat(par,α2,μ2)
# α=ones(8) # order of derivatives
T=215
tSpan=[4,T]

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

## optimazation of μ2 and α2 for integer order model
Data2=LinearInterpolation(data[:,1], data[:,2])

function loss_1(p)# loss function
	# α2, μ2=p[1:2]
	# Par=vcat(par,α2,μ2)
	# α=p[3:10]
	α=p
	if size(x0,2) != Int64(ceil(maximum(α))) # to prevent any errors regarding orders higher than 1
	indx=findall(x-> x>1, α)
	α[indx]=ones(length(indx))
	end
	_, x = FDEsolver(F, tSpan, x0, α, par,h=.01,nc=4)
	S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx=@.I1+R+L+H+B+C-μ3*(N-S-E)
    rmsd(Appx[1:10:end,:], Data2(4:.1:T); normalize=:true) # Normalized root-mean-square error
end

p_lo_1=[.5, .5, .5, .5, .5, .5, .5, .5] #lower bound
p_up_1=[1, 1, 1, 1, 1, 1, 1, 1] # upper bound
p_vec_1=[0.939531849631367, 0.9998860578731849, 0.9997741401361016, 0.9999076787478497, 0.991141852453178, 0.9990686884413122, 0.9432066029272619, 0.7936761619660204]
Res1=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res1=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.99), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(outer_iterations = 10,
						  iterations=300,
						  show_trace=true,
						  show_every=1)
			)

α=vcat(Optim.minimizer(Res1))
α=[1,.9958980293056472,0.8623938517464153,1,1,1,1,1]

T2=250
tSpan2=[200,T2]
_, xx = FDEsolver(F, [4,200], x0, α, par,h=.01,nc=4)
x02=xx[end,:]


function loss_2(p)# loss function
	# α2, μ2=p[1:2]
	# Par=vcat(par,α2,μ2)
	# α=p[3:10]
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
    rmsd(Appx[1:10:end,:], Data2(200:.1:T2); normalize=:true) # Normalized root-mean-square error
end

p_lo_1=[.5, .5, .5, .5, .5, .5, .5, .5] #lower bound
p_up_1=[1, 1, 1, 1, 1, 1, 1, 1] # upper bound
p_vec_1=[0.939531849631367, 0.9998860578731849, 0.9997741401361016, 0.9999076787478497, 0.991141852453178, 0.9990686884413122, 0.9432066029272619, 0.7936761619660204]
Res2=optimize(loss_2,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res1=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.99), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(outer_iterations = 5,
						  iterations=30,
						  show_trace=true,
						  show_every=1)
			)

αα=vcat(Optim.minimizer(Res2))
αα=[0.9624227617441116, 0.9999999994058398, 0.9999999981457667, 0.9999999999999999, 0.9999954941897224, 0.9999999850132952, 0.9999999999465237, 0.9235549807693278]


T3=438
tSpan3=[T2,T3]
# _, xxx = FDEsolver(F, tSpan2, x02, αα, par,h=.01,nc=4)
# x03=xxx[end,:]
_, xxx = FDEsolver(F, [0,250], x0, ones(8), par,h=.01,nc=4)
x03=xxx[end,:]

function loss_3(p)# loss function
	# α2, μ2=p[1:2]
	# Par=vcat(par,α2,μ2)
	# α=p[3:10]
	α=p
	if size(x0,2) != Int64(ceil(maximum(α))) # to prevent any errors regarding orders higher than 1
	indx=findall(x-> x>1, α)
	α[indx]=ones(length(indx))
	end
	_, x = FDEsolver(F, tSpan3, x03, α, par,h=.01,nc=4)
	S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx=@.I1+R+L+H+B+C-μ3*(N-S-E)
    rmsd(Appx[1:10:end,:], Data2(T2:.1:T3); normalize=:true) # Normalized root-mean-square error
end

p_lo_1=[.5, .5, .5, .5, .5, .5, .5, .5] #lower bound
p_up_1=[1, 1, 1, 1, 1, 1, 1, 1] # upper bound
p_vec_1=[0.9999826367080396, 0.583200182055744, 0.9999997090382831, 0.9999552779111426, 0.9999931233485699, 0.999931061595331, 0.9999999960137106, 0.9999997372162407]
Res3=optimize(loss_3,p_lo_1,p_up_1,p_vec_1,Fminbox(LBFGS()),# Broyden–Fletcher–Goldfarb–Shanno algorithm
# Res1=optimize(loss_1,p_lo_1,p_up_1,p_vec_1,SAMIN(rt=.99), # Simulated Annealing algorithm (sometimes it has better perfomance than (L-)BFGS)
			Optim.Options(outer_iterations = 5,
						  iterations=30,
						  show_trace=true,
						  show_every=1)
			)
ααα=vcat(Optim.minimizer(Res3))
ααα=[0.9999826367080396, 0.583200182055744, 0.9999997090382831, 0.9999552779111426, 0.9999931233485699, 0.999931061595331, 0.9999999960137106, 0.9999997372162407]
ααα=ones(8);ααα[2]=0.583200182055744

#plotting

#3 section of fractional orders
t1, x1= FDEsolver(F, [4,200], x0, α, par, nc=4,h=.01)
x02=x1[end,:]
t2, x2= FDEsolver(F, [200,250], x02, αα, par, nc=4,h=.01)
x03=x2[end,:]
t3, x3= FDEsolver(F, [T2,T3], x03, ααα, par, nc=4,h=.01)
x=vcat(x1,x2[2:end,:],x3[2:end,:])
t=vcat(t1,t2[2:end,:],t3[2:end,:])
S=x[:,1]; E=x[:,2]; I1=x[:,3]; R=x[:,4];
L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
N=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
Appx=@.I1+R+L+H+B+C-μ3*(N-S-E)
Err=rmsd(Appx[1:10:end,:], Data2(4:.1:T3); normalize=:true) # Normalized root-mean-square error
plot(t,Appx,xlabel="time")
scatter!(data[:,1],data[:,2])

# 2 sections: first integer until 250 then fractional
t1, x1= FDEsolver(F, [4,250], x0, ones(8), par, nc=4,h=.01)
x02=x1[end,:]
t2, x2= FDEsolver(F, tSpan3, x02, ααα, par, nc=4,h=.01)
xf=vcat(x1,x2[2:end,:])
t=vcat(t1,t2[2:end,:])
S=xf[:,1]; E=xf[:,2]; I=xf[:,3]; R=xf[:,4];
L=xf[:,5]; H=xf[:,6]; B=xf[:,7]; C=xf[:,8];
N=sum(xf,dims=2)
μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
Appx=@.I+R+L+H+B+C-μ3*(N-S-E)
Err=rmsd(Appx[1:10:end,:], Data2(4:.1:T3); normalize=:false) # Normalized root-mean-square error
scatter(data[:,1],data[:,2])
plot!(t,Appx,xlabel="time",lw=2)

#ODE
t, x= FDEsolver(F, [4,T3], x0, ones(8), par, nc=4,h=.01)
S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
Err1=rmsd(Appx1[1:10:end,:], Data2(4:.1:T3); normalize=:false) # Normalized root-mean-square error
plot!(t,Appx1,xlabel="time", legendposition=:right, lw=2)
# scatter!(data[:,1],data[:,2])

#plot population
plot(t,N)
plot!(t,N1,legendposition=:right)

plot(t,S)
plot!(t,S1,legendposition=:right)

plot(t,xf)
plot!(t,x, legend=:false)

# Norm plot
using LinearAlgebra
Mf8=@.abs(Appx[1:10:end,:] - Data2(4:.1:T))
M1=@.abs(Appx1[1:10:end,:] - Data2(4:.1:T))

plot(t[1:10:end,:],Mf8)
plot!(t[1:10:end,:],M1)
