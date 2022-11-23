# plots ODE + FDE, variable and constant N, heatmaps
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

#Define the equation with variable N
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

#eq with constant N
parC=[α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,sum(x0)]
function  Fc(t, x, par)

    α1,σ,γ1,γ2,γ3,ϵ,δ1,δ2,τ,βi,βd,βh,βr,μ1,ξ,N=parC
    S, E, I, R, L, H, B, C=x
	βr=0.7265002737432911
	μ=0.06918229886616623

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

##
T1=250
	T2=438
	tSpan=[4,T2]
	tSpan1=[4,T1]
	tSpan2=[T1,T2]

#solve model 1: ODE
t, x= FDEsolver(F, [4,T2], x0, ones(8), par, nc=4,h=.01)
	S=x[:,1]; E=x[:,2]; I=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx1=@.I+R+L+H+B+C-μ3*(N-S-E)
	Err1=rmsd(Appx1[1:10:end,:], Data2(4:.1:T2)) # Root-mean-square error
print("RMSD(model1)=$Err1")

#solve Eq constant N: model 2
_, x = FDEsolver(Fc, tSpan, x0, ones(8), parC,h=.01,nc=4)
	Sc=x[:,1]; Ec=x[:,2]; Ic=x[:,3]; Rc=x[:,4];
	Lc=x[:,5]; Hc=x[:,6]; Bc=x[:,7]; Cc=x[:,8];
	Nc=sum(x0)
	AppxC=@.Ic+Rc+Lc+Hc+Bc+Cc-μ*(Nc-Sc-Ec)
	ErrC=rmsd(AppxC[1:10:end,:], Data2(4:.1:T2))
print("RMSD(model2)=$ErrC")

# model3: all time fractional
αf=[0.9414876354377308, 0.9999999999997804, 0.999999999999543, 0.9999999999998147, 0.9999999999806818, 0.9999999999981127, 0.9430213976522912, 0.7932329945305198]

t, x= FDEsolver(F, tSpan, x0, αf, par, nc=4,h=.01)
	Sf=x[:,1]; Ef=x[:,2]; If=x[:,3]; Rf=x[:,4];
	Lf=x[:,5]; Hf=x[:,6]; Bf=x[:,7]; Cf=x[:,8];
	Nf=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* Nf, α1 .- α2 .*Nf])
	AppxF=@.If+Rf+Lf+Hf+Bf+Cf-μ3*(Nf-Sf-Ef)
	ErrF=rmsd(AppxF[1:10:end,:], Data2(4:.1:T2)) #
print("RMSD(model3)=$ErrF")

# model4 using 2 sections: first integer until 250 then fractional
#optimized order of derivatives
# α=[0.9999826367080396, 0.583200182055744, 0.9999997090382831, 0.9999552779111426, 0.9999931233485699, 0.999931061595331, 0.9999999960137106, 0.9999997372162407]
α=ones(8);α[2]=0.583200182055744
t1, x1= FDEsolver(F, tSpan1, x0, ones(8), par, nc=4,h=.01)
	x02=x1[end,:]
	t2, x2= FDEsolver(F, tSpan2, x02, α, par, nc=4,h=.01)
	xf=vcat(x1,x2[2:end,:])
	t=vcat(t1,t2[2:end,:])
	Sf2=xf[:,1]; Ef2=xf[:,2]; If2=xf[:,3]; Rf2=xf[:,4];
	Lf2=xf[:,5]; Hf2=xf[:,6]; Bf2=xf[:,7]; Cf2=xf[:,8];
	Nf2=sum(xf,dims=2)
	μ3=mean([μ1 .+ μ2 .* Nf2, α1 .- α2 .*Nf2])
	AppxF2=@.If2+Rf2+Lf2+Hf2+Bf2+Cf2-μ3*(Nf2-Sf2-Ef2)
	ErrF2=rmsd(AppxF2[1:10:end,:], Data2(4:.1:T2)) # root-mean-square error
print("RMSD(model4)=$ErrF2")
# Norm/abs
# Mf8=@.abs(Appx[1:10:end,:] - Data2(4:.1:T2))
# M1=@.abs(Appx1[1:10:end,:] - Data2(4:.1:T2))


#solutions
_, x = FDEsolver(Fc, [4,450], x0, ones(8), parC,h=.01,nc=4)
	Sc=x[:,1]; Ec=x[:,2]; Ic=x[:,3]; Rc=x[:,4];
	Lc=x[:,5]; Hc=x[:,6]; Bc=x[:,7]; Cc=x[:,8];
	Nc=sum(x0)
	AppxC=@.Ic+Rc+Lc+Hc+Bc+Cc-μ*(Nc-Sc-Ec)

t, x= FDEsolver(F, [4,450], x0, ones(8), par, nc=4,h=.01)
	S=x[:,1]; E=x[:,2]; I=x[:,3]; R=x[:,4];
	L=x[:,5]; H=x[:,6]; B=x[:,7]; C=x[:,8];
	N=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N, α1 .- α2 .*N])
	Appx1=@.I+R+L+H+B+C-μ3*(N-S-E)

t, x= FDEsolver(F, [4,450], x0, αf, par, nc=4,h=.01)
	Sf=x[:,1]; Ef=x[:,2]; If=x[:,3]; Rf=x[:,4];
	Lf=x[:,5]; Hf=x[:,6]; Bf=x[:,7]; Cf=x[:,8];
	Nf=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* Nf, α1 .- α2 .*Nf])
	AppxF=@.If+Rf+Lf+Hf+Bf+Cf-μ3*(Nf-Sf-Ef)

t1, x1= FDEsolver(F, tSpan1, x0, ones(8), par, nc=4,h=.01)
	x02=x1[end,:]
	t2, x2= FDEsolver(F, [T1,450], x02, α, par, nc=4,h=.01)
	xf=vcat(x1,x2[2:end,:])
	t=vcat(t1,t2[2:end,:])
	Sf2=xf[:,1]; Ef2=xf[:,2]; If2=xf[:,3]; Rf2=xf[:,4];
	Lf2=xf[:,5]; Hf2=xf[:,6]; Bf2=xf[:,7]; Cf2=xf[:,8];
	Nf2=sum(xf,dims=2)
	μ3=mean([μ1 .+ μ2 .* Nf2, α1 .- α2 .*Nf2])
	AppxF2=@.If2+Rf2+Lf2+Hf2+Bf2+Cf2-μ3*(Nf2-Sf2-Ef2)

##plot dynamics
gr()
scatter(data[:,1],data[:,2], label="Real data", c="khaki3",markerstrokewidth=0)
	plot!(t,AppxC,ylabel="Cumulative confirmed cases",lw=2, label="Model1, RMSD=2085", c="darkorange3", linestyle=:dashdot)
	plot!(t,Appx1,lw=2, label="Model2, RMSD=276.5", c="deepskyblue1")
	plot!(t,AppxF,lw=2, label="Model3, RMSD=245.9", c="deeppink", linestyle=:dash)
	plotModels=plot!(t,AppxF2,xlabel="Days", legendposition=:bottomright, lw=2, label="Model4, RMSD=211.0", linestyle=:dot, c="blue1",
	title = "(a)", titleloc = :left, titlefont = font(10))

#plot population
plot(t,Nc*ones(length(t)),c="darkorange3",lw=2,legendposition=:right,label="Model1", linestyle=:dashdot)
	plot!(t,N,lw=2,legendposition=:right,label="Model2", c="deepskyblue1")
	plot!(t,Nf,lw=2,legendposition=:right,label="Model3",linestyle=:dash,c="deeppink")
	plotN=plot!(t,Nf2,label="Model4",linestyle=:dot, lw=2, xlabel="Days", ylabel="Variable population",c="blue1",
	title = "(b)", titleloc = :left, titlefont = font(10))



##plot heatmaps

#optimized order of derivatives
tSpan1=[4,438]

Err=zeros(8,40)
Errr=zeros(8,40)
#FDE: model 3
for i=1:8
    print(i)# to check in which order is counting
    for j=1:40
        print(j)# to check in which itteration is counting
        α=ones(8)
        α[i]=1-j*.01
        t, x= FDEsolver(F, [4,438], x0, α, par, nc=2,h=.01)
        S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
        L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
        N1=sum(x,dims=2)
        μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
        Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
        Err[i,j]=rmsd(Appx1[1:10:end,:], Data2(4:.1:438)) # Normalized root-mean-square error
    end
end

#ODE+FDE: model 4
for i=1:8
    print(i)# to check in which order is counting
    for j=1:40
        print(j)# to check in which itteration is counting
        α=ones(8)
        t1, x1= FDEsolver(F, [4,250], x0, α, par, nc=2,h=.01)
        x02=x1[end,:]
        α[i]=1-j*.01
        t2, x2= FDEsolver(F, [250,438], x02, α, par, nc=2,h=.01)
        x=vcat(x1,x2[2:end,:])
        t=vcat(t1,t2[2:end,:])
        S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
        L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
        N1=sum(x,dims=2)
        μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
        Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
        Errr[i,j]=rmsd(Appx1[1:10:end,:], Data2(4:.1:438)) # Normalized root-mean-square error
    end
end

RMSD1= log10(Err1) #error for integer order

# plot heatmap model 4
CoLor = cgrad([:yellow2,:darkgoldenrod1,:mediumpurple], [.45,.4], categorical = true) # define color
heatmap(["S","E","I","R","L","H","B","C"],0.6:.01:.99,log10.(Errr[:,end:-1:1]'),
        color=CoLor,colorbar_title="log10(RMSD)",clim=(RMSD1-.18,RMSD1+.21),
        xlabel="Individual classes", ylabel="Order of derivative",
		title = "(e) Model4 ", titleloc = :left, titlefont = font(10))
		pltheat=plot!(Shape(0 .+ [1,2,2,1], 0 .+ [.595,.595,.995,.995]), fillcolor=:false, legend=:false)


# plot heatmap model 4
pltheat1=heatmap(["S","E","I","R","L","H","B","C"],0.6:.01:.99,log10.(Err[:,end:-1:1]'),
        color=CoLor,colorbar_title="log10(RMSD)",clim=(RMSD1-.18,RMSD1+.21),
        xlabel="Individual classes", ylabel="Order of derivative",
		title = "(d) Model3 ", titleloc = :left, titlefont = font(10), colorbar=:false)


# savefig(pltheat,"pltheat.svg")
# savefig(pltheat1,"pltheat1.svg")


tt, x= FDEsolver(F, [4,438], x0, ones(8), par, nc=2,h=.01)
	S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
	L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
	N1=sum(x,dims=2)
	μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
	Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
	# rmsd(Appx1[1:10:end,:], Data2(4:.1:438))


#plot
plot()
N=30
err=zeros(N)
for i=1:N
            α=ones(8)
            t1, x1= FDEsolver(F, [4,250], x0, α, par, nc=2,h=.01)
            x02=x1[end,:]
            α[2]=1-i*.02
            t2, x2= FDEsolver(F, [250,438], x02, α, par, nc=2,h=.01)
            x=vcat(x1,x2[2:end,:])
            t=vcat(t1,t2[2:end,:])
            S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
            L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
            N1=sum(x,dims=2)
            μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
            Appx=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
            # err[i]=copy(rmsd(Appx[1:10:end,:], Data2(4:.1:438)))
            plot!(t[25000:100:end],Appx[25000:100:end,:],lw=2; alpha=0.2, color="blue1")

end

scatter!(data[74:end,1],data[74:end,2], label="Real data", c="khaki3",markerstrokewidth=0)
pltFtry=plot!(tt[25000:50:end],Appx1[25000:50:end,:],xlabel="Days",ylabel="Cumulative confirmed cases",
        lw=2, c="chocolate1",label="Integer model",legend=:false,
		title = "(c)", titleloc = :left, titlefont = font(10))

# savefig(pltFtry,"pltFtry.svg")


l = @layout [a{0.5h}; [grid(1,2)]; [grid(1,1) b{0.6w}]]
Allplots=plot(plotModels, plotN, pltFtry, pltheat1, pltheat, layout= l, size = (900, 900))

savefig(Allplots,"Allplots.svg")
