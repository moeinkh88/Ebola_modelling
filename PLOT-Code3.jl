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
tSpan1=[4,438]

Err1=zeros(8,40)
#FDE
for i=1:8
    print(i)
    for j=1:40
        print(j)
        α=ones(8)
        α[i]=1-j*.01
        t, x= FDEsolver(F, [4,438], x0, α, par, nc=2,h=.01)
        S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
        L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
        N1=sum(x,dims=2)
        μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
        Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
        Err1[i,j]=rmsd(Appx1[1:10:end,:], Data2(4:.1:438)) # Normalized root-mean-square error
    end
end

#ODE+FDE
for i=1:8
    print(i)
    for j=1:40
        print(j)
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
        Err1[i,j]=rmsd(Appx1[1:10:end,:], Data2(4:.1:438)) # Normalized root-mean-square error
    end
end

# Err=Err1 .- 276.5100875995236

backend(:plotly)
pltheat=heatmap(["S","E","I","R","L","H","B","C"],0.6:.01:.99,log10.(Err1[:,end:-1:1]'),
        c=:balance,colorbar_title="log10(RMSD)",
        xlabel="Individual classes", ylabel="Order of derivative")

savefig(pltheat,"pltheat.svg")

t, x= FDEsolver(F, [4,438], x0, ones(8), par, nc=2,h=.01)
S1=x[:,1]; E1=x[:,2]; I1=x[:,3]; R1=x[:,4];
L1=x[:,5]; H1=x[:,6]; B1=x[:,7]; C1=x[:,8];
N1=sum(x,dims=2)
μ3=mean([μ1 .+ μ2 .* N1, α1 .- α2 .*N1])
Appx1=@.I1+R1+L1+H1+B1+C1-μ3*(N1-S1-E1)
rmsd(Appx1[1:10:end,:], Data2(4:.1:438))


#plot


# plot(; legend=:false)
scatter(data[74:end,1],data[74:end,2], label="Real data", c="khaki3",markerstrokewidth=0)
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
            plot!(t[25000:20:end],Appx[25000:20:end,:],lw=2; alpha=0.2, color="blue1")

end

# plot!(t[25000:end],Appx1[25000:end,:],ylabel="Cumulative confirmed cases",
    # lw=2, c="darkorange3",label="Integer model",legendposition=:bottomright)
pltFtry=plot!(t[25000:20:end],Appx1[25000:20:end,:],xlabel="Days",ylabel="Cumulative confirmed cases",
        lw=2, c="darkorange3",label="Integer model",legend=:false)


savefig(pltFtry,"pltFtry.svg")
