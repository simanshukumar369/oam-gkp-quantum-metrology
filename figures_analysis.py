"""
figures_analysis.py  
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfc

os.makedirs("results/figures", exist_ok=True)

plt.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "mathtext.fontset":"stix","font.size":8,"axes.labelsize":8,
    "axes.titlesize":8,"xtick.labelsize":7,"ytick.labelsize":7,
    "legend.fontsize":7,"axes.linewidth":0.8,
    "axes.spines.top":False,"axes.spines.right":False,
    "xtick.direction":"out","ytick.direction":"out",
    "xtick.major.size":3,"ytick.major.size":3,
    "lines.linewidth":1.3,"savefig.dpi":300,"pdf.fonttype":42,
})

COL = {"curve":"#333333","sq":"#2166AC","opt":"#1A6B2A",
       "routeA":"#C0392B","routeB":"#8E44AD","cap":"#B5651D",
       "thresh":"#E74C3C"}
DOUBLE = 183/25.4
A = np.sqrt(2*np.pi)

def Q(x): return 0.5*erfc(x/np.sqrt(2))
def phi(x): return np.exp(-0.5*x**2)/np.sqrt(2*np.pi)

def sigma_q(th,eta,gamma): return np.sqrt((1-eta)/(2*eta)+gamma*np.sin(th)**2)
def sigma_p(th,eta,gamma): return np.sqrt((1-eta)/(2*eta)+gamma*np.cos(th)**2)

def perr_fn(th,r=1.092,eta=0.9,gamma=0.05):
    sq=sigma_q(th,eta,gamma); sp=sigma_p(th,eta,gamma)
    Pq=2*Q(A*r/(2*sq)); Pp=2*Q(A/r/(2*sp))
    return Pq+Pp-Pq*Pp

def cap_fn(th,qfi=9.764,**kw):
    p=perr_fn(th,**kw)
    return qfi*(-np.log(p)) if p>0 else np.nan

def theta_star(eta,gamma,r=1.092):
    def bal(th):
        sq=sigma_q(th,eta,gamma); sp=sigma_p(th,eta,gamma)
        uq=A*r/(2*sq); up=A/r/(2*sp)
        return r**2*phi(uq)/sq**3-phi(up)/sp**3
    try:
        fa=bal(0.02); fb=bal(np.pi/2-0.02)
        if fa*fb<0: return brentq(bal,0.02,np.pi/2-0.02)
    except: pass
    return minimize_scalar(lambda t:perr_fn(t,r=r,eta=eta,gamma=gamma),
                           bounds=(0.02,np.pi/2-0.02),method='bounded').x

# ─── FIG A ───────────────────────────────────────────────────────────────────
def figure_A():
    ETA,GAMMA,R = 0.9,0.05,1.092
    th_arr  = np.linspace(0,np.pi/2,600)
    th_deg  = np.degrees(th_arr)
    perr_arr= np.array([perr_fn(t,R,ETA,GAMMA) for t in th_arr])
    cap_arr = np.array([cap_fn(t,eta=ETA,gamma=GAMMA) for t in th_arr])

    ell_all=[0.0,0.5,1.0,1.5,2.0,2.5,3.0]; ell_max=4.0
    th_ell  =[min(e*180/ell_max,180-e*180/ell_max) for e in ell_all]
    perr_ell=[perr_fn(np.radians(t),R,ETA,GAMMA) for t in th_ell]
    cap_ell =[cap_fn(np.radians(t),eta=ETA,gamma=GAMMA) for t in th_ell]
    is_int  =[abs(e-round(e))<0.01 for e in ell_all]

    th_star_deg = np.degrees(theta_star(ETA,GAMMA,R))
    th_A,th_B   = 67.5,60.0
    P_sq = perr_fn(0.0,R,ETA,GAMMA)
    C_sq = cap_fn(0.0,eta=ETA,gamma=GAMMA)

    # Three columns: panel a, panel b, legend
    fig = plt.figure(figsize=(DOUBLE,2.6))
    gs  = fig.add_gridspec(1,2,width_ratios=[1,1],wspace=0.50)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Panel a ──────────────────────────────────────────────────────────────
    ax=ax1
    ax.axhspan(1e-9,1e-3,alpha=0.10,color="#27AE60",zorder=0)
    ax.axhline(1e-3,color=COL["thresh"],lw=0.8,ls="--")
    # P_th label — top right of fault-tolerant zone
    ax.text(89,0.8e-3,r"$P_\mathrm{th}$",color=COL["thresh"],
            fontsize=6.5,ha="right",va="bottom")

    ax.axhline(P_sq,color=COL["sq"],lw=0.7,ls=":",alpha=0.6)
    # Square baseline label — at left, not overlapping P_th
    ax.text(1,P_sq*1.06,"Square baseline",color=COL["sq"],fontsize=5.8,
            ha="left",va="bottom",alpha=0.8)

    ax.semilogy(th_deg,perr_arr,color=COL["curve"],lw=1.6,zorder=4)

    # Vertical lines
    for th,col,ls in [(th_star_deg,COL["opt"],"-."),(th_A,COL["routeA"],"--"),(th_B,COL["routeB"],"--")]:
        ax.axvline(th,color=col,lw=0.85,ls=ls,alpha=0.85,zorder=3)

    # Discrete markers
    for th,p,ii,ell in zip(th_ell,perr_ell,is_int,ell_all):
        col=COL["sq"] if ell==0 else COL["routeA"] if ell==1.5 else "#777777"
        ax.semilogy(th,p,"o" if ii else "D",color=col,
                    ms=6 if ii else 5,mec="white",mew=0.6,zorder=8)

    # Key marker labels — only ell=0, 1.5, 2.0, placed to avoid overlap
    ax.text(2.5,P_sq*0.68,r"$\ell=0$",color=COL["sq"],fontsize=5.5,ha="left")
    ax.text(th_ell[3]+1.0,perr_ell[3]*1.5,r"$\ell=1.5$",
            color=COL["routeA"],fontsize=5.5,ha="left")
    ax.text(th_ell[4]+1.0,perr_ell[4]*0.55,r"$\ell=2$",
            color="#777777",fontsize=5.5,ha="right")

    ax.set_xlabel(r"Lattice rotation $\theta$ (degrees)")
    ax.set_ylabel(r"Logical error rate $P_\mathrm{err}$")
    ax.set_title(rf"$P_\mathrm{{err}}(\theta)$   ($\eta={ETA}$, $\gamma={GAMMA}$)",fontsize=8,pad=4)
    ax.set_xlim(-2,92); ax.set_ylim(6e-6,8e-4)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.text(0.04,0.99,"a",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top")

    # ── Panel b ──────────────────────────────────────────────────────────────
    ax=ax2
    ax.plot(th_deg,cap_arr,color=COL["cap"],lw=1.6,zorder=4)
    for th,col,ls in [(th_star_deg,COL["opt"],"-."),(th_A,COL["routeA"],"--"),(th_B,COL["routeB"],"--")]:
        ax.axvline(th,color=col,lw=0.85,ls=ls,alpha=0.85,zorder=3)
    ax.axhline(C_sq,color=COL["sq"],lw=0.7,ls=":",alpha=0.6)

    for th,c,ii,ell in zip(th_ell,cap_ell,is_int,ell_all):
        col=COL["sq"] if ell==0 else COL["routeA"] if ell==1.5 else "#777777"
        ax.plot(th,c,"o" if ii else "D",color=col,
                ms=6 if ii else 5,mec="white",mew=0.6,zorder=8)

    C_max=cap_fn(np.radians(th_A),eta=ETA,gamma=GAMMA)
    # Max annotation — placed left of the peak, not on top of it
    ax.annotate(rf"max $\mathcal{{C}}={C_max:.1f}$",
                xy=(th_A,C_max),xytext=(70,C_max-7),fontsize=5.8,
                color=COL["routeA"],
                arrowprops=dict(arrowstyle="->",color=COL["routeA"],lw=0.7))

    # +41% at midpoint of the arrow
    ax.annotate("",xy=(th_star_deg,C_max),xytext=(0,C_sq),
                arrowprops=dict(arrowstyle="->",color=COL["opt"],lw=0.9))
    ax.text((th_star_deg)/2-1,(C_sq+C_max)/2,"+41%",fontsize=6.0,
            color=COL["opt"],ha="center",va="center",
            bbox=dict(fc="white",ec="none",pad=1))

    ax.set_xlabel(r"Lattice rotation $\theta$ (degrees)")
    ax.set_ylabel(r"Capacity $\mathcal{C}=\mathcal{F}_Q(-\ln P_\mathrm{err})$")
    ax.set_title(rf"Capacity $\mathcal{{C}}(\theta)$   ($\eta={ETA}$, $\gamma={GAMMA}$)",fontsize=8,pad=4)
    ax.set_xlim(-2,92)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.text(0.04,0.99,"b",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top")

    # ── Legend inside panel a, bottom right ──────────────────────────────────
    from matplotlib.lines import Line2D; from matplotlib.patches import Patch
    h=[Line2D([0],[0],color=COL["curve"],lw=1.5,label="Analytic"),
       Line2D([0],[0],color="k",marker="o",ms=5,lw=0,mec="white",label=r"Integer $\ell$"),
       Line2D([0],[0],color="k",marker="D",ms=4,lw=0,mec="white",label=r"Fractional $\ell$"),
       Line2D([0],[0],color=COL["opt"],   lw=0.9,ls="-.",label=rf"$\theta^*={th_star_deg:.1f}^\circ$"),
       Line2D([0],[0],color=COL["routeA"],lw=0.9,ls="--",label=r"Route A ($\ell=1.5$)"),
       Line2D([0],[0],color=COL["routeB"],lw=0.9,ls="--",label=r"Route B ($\ell=2$, $\ell_{{\rm max}}=6$)"),
       Patch(fc="#27AE60",alpha=0.15,ec="none",label="Fault-tolerant"),]
    ax1.legend(handles=h,loc="lower left",frameon=True,fontsize=5.8,
               handlelength=1.6,labelspacing=0.40,borderpad=0.5,
               framealpha=0.92,edgecolor="#cccccc",fancybox=False)

    for p in ["results/figures/fig_A_perr_theta_curve.pdf",
              "results/figures/fig_A_perr_theta_curve.png"]:
        fig.savefig(p,bbox_inches="tight",pad_inches=0.02)
    plt.close(); print("  Saved Fig A")


# ─── FIG B ───────────────────────────────────────────────────────────────────
def figure_B():
    eta_arr  = np.linspace(0.70,0.99,130)
    gam_arr  = np.linspace(0.005,0.25,130)
    ETA_G,GAMMA_G = np.meshgrid(eta_arr,gam_arr)
    LOSS_ARR = 1-eta_arr

    print("  Computing theta* grid (130x130)...")
    THETA  = np.zeros_like(ETA_G)
    IMPROV = np.zeros_like(ETA_G)
    for i in range(ETA_G.shape[0]):
        for j in range(ETA_G.shape[1]):
            e,g   = ETA_G[i,j],GAMMA_G[i,j]
            th    = theta_star(e,g)
            THETA[i,j]  = np.degrees(th)
            P_opt = perr_fn(th,eta=e,gamma=g)
            P_sq  = perr_fn(0,eta=e,gamma=g)
            IMPROV[i,j] = np.log10(max(P_sq/max(P_opt,1e-15),1.001))

    fig,axes=plt.subplots(1,2,figsize=(DOUBLE,2.7),gridspec_kw={"wspace":0.44})

    # ── Panel a ──────────────────────────────────────────────────────────────
    ax=axes[0]
    cmap_a=matplotlib.colormaps["RdYlBu_r"].resampled(18)
    norm_a=BoundaryNorm(np.linspace(45,90,19),ncolors=cmap_a.N)
    ax.pcolormesh(LOSS_ARR,gam_arr,THETA,cmap=cmap_a,norm=norm_a,
                  shading="gouraud",rasterized=True)

    # Contours at 4 levels
    cont_lvl=[60,64.4,67.5,75]
    cont_col=["#8E44AD","#1A6B2A","#C0392B","#555555"]
    cont_sty=["--","-.","--",":"]
    ax.contour(LOSS_ARR,gam_arr,THETA,levels=cont_lvl,
               colors=cont_col,linewidths=[0.9,1.0,0.9,0.7],
               linestyles=cont_sty)

    # Labels at right side, staggered y positions
    label_info=[
        (60,  "#8E44AD", r"$60^\circ$ (B)",          0.230, 0.028),
        (64.4,"#1A6B2A", r"$\theta^*=64.4^\circ$",   0.225, 0.050),
        (67.5,"#C0392B", r"$67.5^\circ$ (A)",         0.220, 0.072),
        (75,  "#555555", r"$75^\circ$",               0.225, 0.105),
    ]
    for _,col,lbl,x,y in label_info:
        ax.text(x,y,lbl,color=col,fontsize=5.5,ha="right",va="center",
                bbox=dict(fc="white",ec="none",pad=0.5,alpha=0.85))

    # Simulation points with labels safely placed
    ax.plot(0.10,0.05,"o",color="white",ms=7,mec="black",mew=0.8,zorder=10)
    ax.plot(0.20,0.10,"^",color="white",ms=7,mec="black",mew=0.8,zorder=10)
    ax.text(0.10+0.005,0.05+0.012,r"$(0.9,0.05)$"+"\n"+r"$64.4^\circ$",
            fontsize=4.8,ha="left",va="bottom",
            bbox=dict(fc="white",ec="none",alpha=0.8,pad=0.5))
    ax.text(0.20+0.005,0.10+0.012,r"$(0.8,0.1)$"+"\n"+r"$71.5^\circ$",
            fontsize=4.8,ha="left",va="bottom",
            bbox=dict(fc="white",ec="none",alpha=0.8,pad=0.5))

    im_a=matplotlib.cm.ScalarMappable(norm=norm_a,cmap=cmap_a)
    cb=fig.colorbar(im_a,ax=ax,pad=0.02,shrink=0.95)
    cb.set_label(r"$\theta^*$ (degrees)",fontsize=7)
    cb.set_ticks([45,55,65,75,85,90])
    cb.ax.tick_params(labelsize=6)

    ax.set_xlabel(r"Loss rate $1-\eta$",fontsize=7)
    ax.set_ylabel(r"Dephasing rate $\gamma$",fontsize=7)
    ax.set_title(r"Optimal rotation $\theta^*(\eta,\gamma)$",fontsize=8,pad=4)
    ax.tick_params(labelsize=6)
    ax.text(0.02,0.97,"a",transform=ax.transAxes,fontsize=9,
            fontweight="bold",va="top",color="black")

    # ── Panel b ──────────────────────────────────────────────────────────────
    ax=axes[1]
    cmap_b=matplotlib.colormaps["YlOrRd"].resampled(20)
    im2=ax.pcolormesh(1-eta_arr,gam_arr,IMPROV,cmap=cmap_b,
                      vmin=0,vmax=2,shading="gouraud",rasterized=True)
 
    # Contours at 2×,5×,10×,20× with auto-placed inline rotated labels
    cs2=ax.contour(1-eta_arr,gam_arr,IMPROV,
                   levels=[np.log10(v) for v in [2,5,10,20]],
                   colors=["#555","#333","#111","white"],
                   linewidths=[0.8,0.9,1.0,1.1])
    fmt2={np.log10(v): f"{v}\u00d7" for v in [2,5,10,20]}
    ax.clabel(cs2,fmt=fmt2,fontsize=5.5,inline=True)
 
    # Simulation points
    ax.plot(0.10,0.05,"o",color="white",ms=7,mec="black",mew=0.8,zorder=10)
    ax.plot(0.20,0.10,"^",color="white",ms=7,mec="black",mew=0.8,zorder=10)
 
    cb2=fig.colorbar(im2,ax=ax,pad=0.02,shrink=0.95,
                     ticks=[0,np.log10(2),np.log10(5),1,np.log10(20),np.log10(50),2])
    cb2.ax.set_yticklabels(["1\u00d7","2\u00d7","5\u00d7","10\u00d7",
                             "20\u00d7","50\u00d7","100\u00d7"],fontsize=6)
    cb2.set_label(r"Improv. vs square ($\log_{10}$ scale)",fontsize=7)
 
    ax.set_xlabel(r"Loss rate $1-\eta$",fontsize=7)
    ax.set_ylabel(r"Dephasing rate $\gamma$",fontsize=7)
    ax.set_title(r"$P_\mathrm{err}$ improvement of $\theta^*$ over square",
                 fontsize=8,pad=4)
    ax.tick_params(labelsize=6)
    ax.text(0.01,0.97,"b",transform=ax.transAxes,fontsize=9,fontweight="bold",va="top")
 
    for p in ["results/figures/fig_B_theta_phase_diagram.pdf",
              "results/figures/fig_B_theta_phase_diagram.png"]:
        fig.savefig(p,bbox_inches="tight",pad_inches=0.02)
    plt.close(); print("  Saved Fig B")
 
 

if __name__=="__main__":
    print("\nGenerating Fig A and Fig B...")
    figure_A()
    figure_B()
    print("Done.")
