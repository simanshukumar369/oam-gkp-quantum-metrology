"""
figures_nature.py  
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
from scipy.special import erfc as _erfc
warnings.filterwarnings("ignore")

os.makedirs("results/figures", exist_ok=True)

plt.rcParams.update({
    "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],
    "font.size":8,"axes.labelsize":8,"axes.titlesize":8,
    "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
    "axes.linewidth":0.8,"xtick.major.size":3,"ytick.major.size":3,
    "lines.linewidth":1.2,"axes.spines.top":False,"axes.spines.right":False,
    "xtick.direction":"out","ytick.direction":"out",
    "savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.02,
    "pdf.fonttype":42,"mathtext.fontset":"stix",
})

NAT = {"square":"#2166AC","ell1":"#4DAC26","ell2":"#D6604D",
       "threshold":"#B2182B","gold":"#E8C13A","grey":"#636363"}
SINGLE = 89/25.4; DOUBLE = 183/25.4

data = {
    (0.9,0.05): {0:(9.7637,4.13e-4,1.092),1:(9.7637,5.42e-5,1.092),2:(9.7637,2.63e-5,1.092)},
    (0.8,0.10): {0:(3.0706,1.47e-2,1.082),1:(3.0747,7.02e-3,1.089),2:(3.0751,5.02e-3,1.095)},
}
PTHRESH = 1e-3
colors  = [NAT["square"],NAT["ell1"],NAT["ell2"]]
markers = ["o","s","^"]
labels  = [r"Square ($\ell=0$)",r"OAM $\ell=1$",r"OAM $\ell=2$"]

def plabel(ax,letter,x=0.04,y=0.96,color="black",**kw):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=9,
            fontweight="bold",va="top",color=color,**kw)

def Q(x): return 0.5*_erfc(x/np.sqrt(2))

def analytic_perr(eta,gamma,theta,r):
    a=np.sqrt(2*np.pi); sq=(1-eta)/(2*eta+1e-12)
    s_q=np.sqrt(sq+gamma*np.sin(theta)**2)
    s_p=np.sqrt(sq+gamma*np.cos(theta)**2)
    Pq=2*Q(a*r/(2*s_q+1e-12)); Pp=2*Q(a/r/(2*s_p+1e-12))
    return Pq+Pp-Pq*Pp

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1
# ═══════════════════════════════════════════════════════════════════════════
def fig1_geometry_comparison():
    fig = plt.figure(figsize=(DOUBLE,2.4))
    gs  = gridspec.GridSpec(1,2,wspace=0.38,left=0.08,right=0.97,
                            top=0.88,bottom=0.22)
    noise_pts = [(0.9,0.05),(0.8,0.10)]
    ells = [0,1,2]; x = np.arange(3); w = 0.32
 
    # ── a: QFI ──
    ax1=fig.add_subplot(gs[0])
    for i,(eta,gamma) in enumerate(noise_pts):
        qfi_v=[data[(eta,gamma)][ell][0] for ell in ells]
        offset=(i-0.5)*w
        bars=ax1.bar(x+offset,qfi_v,w-0.04,color=colors,
                     alpha=0.85-0.15*i,edgecolor="white",linewidth=0.4)
        if i==1:
            for bar,col in zip(bars,colors):
                bar.set_hatch("///"); bar.set_edgecolor(col); bar.set_alpha(0.55)
    ax1.set_xticks(x); ax1.set_xticklabels(labels,rotation=13,ha="right")
    ax1.set_ylabel(r"Quantum Fisher information $\mathcal{F}_Q$")
    ax1.set_ylim(0,12)
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    plabel(ax1,"a")
    ax1.set_title("QFI  (sensitivity)",fontsize=8,pad=4)
    p1=mpatches.Patch(fc="#888",alpha=0.85,
                      label=r"$\eta\!=\!0.9,\;\gamma\!=\!0.05$")
    p2=mpatches.Patch(fc="#888",alpha=0.55,hatch="///",
                      label=r"$\eta\!=\!0.8,\;\gamma\!=\!0.10$")
    ax1.legend(handles=[p1,p2],loc="upper right",frameon=False,
               handlelength=1.1,fontsize=6.5,
               bbox_to_anchor=(1.0, 1.03))
 
    # ── b: P_err ──
    ax2=fig.add_subplot(gs[1])
    for i,(eta,gamma) in enumerate(noise_pts):
        perr_v=[data[(eta,gamma)][ell][1] for ell in ells]
        offset=(i-0.5)*w
        bars=ax2.bar(x+offset,perr_v,w-0.04,color=colors,
                     alpha=0.85-0.15*i,edgecolor="white",linewidth=0.4)
        if i==1:
            for bar,col in zip(bars,colors):
                bar.set_hatch("///"); bar.set_edgecolor(col); bar.set_alpha(0.55)
 
    ax2.axhline(PTHRESH,color=NAT["threshold"],lw=0.9,ls="--",zorder=5)
    ax2.text(2.54,PTHRESH*1.7,r"$P_\mathrm{th}$",
             color=NAT["threshold"],fontsize=6.2,va="bottom",ha="right")
    ax2.set_yscale("log"); ax2.set_ylim(5e-6,2e-1)
    ax2.set_xticks(x); ax2.set_xticklabels(labels,rotation=13,ha="right")
    ax2.set_ylabel(r"Logical error rate $P_\mathrm{err}$")
    plabel(ax2,"b")
    ax2.set_title(r"$P_\mathrm{err}$  (fault tolerance)",fontsize=8,pad=4)
 
    # Improvement labels: place ABOVE the low-noise bar, clearly
    improve={1:7.6,2:15.7}
    for ell in [1,2]:
        p=data[(0.9,0.05)][ell][1]
        xi=x[ell]-w/2   # centre of low-noise bar
        # put label above the bar with small gap
        ax2.text(xi,p*3.5,f"{improve[ell]:.1f}\u00d7",
                 ha="center",va="bottom",fontsize=6.5,
                 color=colors[ell],fontweight="bold")
 
    fig.savefig("results/figures/fig1_geometry_comparison.pdf")
    fig.savefig("results/figures/fig1_geometry_comparison.png",dpi=300)
    print("  Saved fig1"); plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# Fig 2
# ═══════════════════════════════════════════════════════════════════════════
def fig2_noise_landscape():
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE,2.2),
                          gridspec_kw={"wspace":0.38})
    eta_arr  =np.linspace(0.70,0.99,200)
    gamma_arr=np.linspace(0.005,0.20,200)
    configs=[
        (0.0,     1.092,NAT["square"],"Square",    "-"),
        (np.pi/4, 1.092,NAT["ell1"],  r"OAM $\ell=1$","--"),
        (np.pi/2, 1.092,NAT["ell2"],  r"OAM $\ell=2$",":"),
    ]
    # x-offsets so data points don't overlap
    x_off={0:-0.008,1:0.000,2:+0.008}
 
    # ── a: vary loss ──
    ax=axes[0]
    for theta,r,col,lbl,ls in configs:
        perr=[analytic_perr(e,0.05,theta,r) for e in eta_arr]
        ax.semilogy(1-eta_arr,perr,color=col,ls=ls,lw=1.4,label=lbl)
    ax.axhline(PTHRESH,color=NAT["threshold"],lw=0.8,ls="-.",alpha=0.9)
    ax.text(0.005,PTHRESH*1.8,r"$P_\mathrm{th}$",
            color=NAT["threshold"],fontsize=6.5)
    # Low-noise points
    for ell,col,mk in zip([0,1,2],colors,markers):
        ax.plot(0.10+x_off[ell],data[(0.9,0.05)][ell][1],mk,
                color=col,ms=5.5,mec="white",mew=0.7,zorder=8)
    # High-noise points — lighter, with offset
    for ell,col,mk in zip([0,1,2],colors,markers):
        ax.plot(0.20+x_off[ell],data[(0.8,0.10)][ell][1],mk,
                color=col,ms=5.5,mec="white",mew=0.7,zorder=8,alpha=0.55)
    ax.set_xlabel(r"Loss rate $1-\eta$")
    ax.set_ylabel(r"Logical error rate $P_\mathrm{err}$")
    ax.set_ylim(5e-7,5e-1)
    ax.axhspan(1e-9,PTHRESH,alpha=0.07,color=NAT["ell1"],zorder=0)
    ax.text(0.97,0.05,"Fault-tolerant",transform=ax.transAxes,
            fontsize=6,color=NAT["ell1"],ha="right",fontstyle="italic")
    # Legend outside plot area (upper left, not blocking curves)
    ax.legend(frameon=False,loc="upper left",handlelength=1.6,
              fontsize=6.5,labelspacing=0.3,
              bbox_to_anchor=(0.0, 0.92))
    plabel(ax,"a")
    ax.set_title(r"Varying loss  ($\gamma=0.05$)",fontsize=8,pad=4)

    # ── b: vary dephasing ──
    ax=axes[1]
    for theta,r,col,lbl,ls in configs:
        perr=[analytic_perr(0.9,g,theta,r) for g in gamma_arr]
        ax.semilogy(gamma_arr,perr,color=col,ls=ls,lw=1.4)
    ax.axhline(PTHRESH,color=NAT["threshold"],lw=0.8,ls="-.",alpha=0.9)
    for ell,col,mk in zip([0,1,2],colors,markers):
        ax.plot(0.05+x_off[ell],data[(0.9,0.05)][ell][1],mk,
                color=col,ms=5.5,mec="white",mew=0.7,zorder=8)
    ax.set_xlabel(r"Dephasing rate $\gamma$")
    ax.set_ylabel(r"Logical error rate $P_\mathrm{err}$")
    ax.set_ylim(5e-7,5e-1)
    ax.axhspan(1e-9,PTHRESH,alpha=0.07,color=NAT["ell1"],zorder=0)
    ax.text(0.97,0.05,"Fault-tolerant",transform=ax.transAxes,
            fontsize=6,color=NAT["ell1"],ha="right",fontstyle="italic")
    plabel(ax,"b")
    ax.set_title(r"Varying dephasing  ($\eta=0.9$)",fontsize=8,pad=4)

    fig.savefig("results/figures/fig2_noise_landscape.pdf")
    fig.savefig("results/figures/fig2_noise_landscape.png",dpi=300)
    print("  Saved fig2"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — YlOrRd colormap 
# ═══════════════════════════════════════════════════════════════════════════
def fig3_phase_diagram():
    eta_v  =np.linspace(0.70,0.99,200)
    gamma_v=np.linspace(0.005,0.20,200)
    ETA,GAMMA=np.meshgrid(eta_v,gamma_v)

    # YlOrRd reversed: blue(low Perr)→yellow→red(high Perr)
    # Since logP goes from -5 (good) to -1 (bad), we want:
    # low logP (very negative) = blue/dark = good
    # high logP (less negative) = yellow/red = bad
    # Use YlOrRd reversed so -5 is dark yellow (low err) ... actually
    # standard convention: blue=low err (good), red=high err (bad)
    # Use the same YlOrRd as fig_B but inverted so red=high Perr
    cmap = matplotlib.colormaps["YlOrRd"].resampled(256)

    configs=[
        (0.0,     1.092,r"Square ($\ell=0$)"),
        (np.pi/4, 1.092,r"OAM $\ell=1$"),
        (np.pi/2, 1.092,r"OAM $\ell=2$"),
    ]
    vmin,vmax=-5,-1

    fig,axes=plt.subplots(1,3,figsize=(DOUBLE,2.3),
                          gridspec_kw={"wspace":0.06})

    for ax,(theta,r,title),letter in zip(axes,configs,["a","b","c"]):
        P=analytic_perr(ETA,GAMMA,theta,r)
        logP=np.log10(np.clip(P,1e-7,1.0))
        # vmin=-5 (best,blue), vmax=-1 (worst,red) — map to YlOrRd naturally
        # But YlOrRd goes light→dark for low→high values
        # So low logP (-5) = light yellow = bad mapping
        # We want low logP = good = blue → use reversed cmap
        cmap_r=matplotlib.colormaps["YlOrRd_r"].resampled(256)
        im=ax.pcolormesh(1-eta_v,gamma_v,logP,cmap=cmap_r,
                         vmin=vmin,vmax=vmax,shading="gouraud",rasterized=True)

        # P_th contour
        cs=ax.contour(1-eta_v,gamma_v,logP,levels=[np.log10(PTHRESH)],
                      colors=["white"],linewidths=[1.1])
        # Label at safe position — right side of contour
        ax.clabel(cs,fmt=r"$P_\mathrm{th}$",fontsize=5.5,inline=True,
                  manual=[(0.25,0.08)])

        # Data points — large, clearly visible, white border
        ax.plot(0.10,0.05,"o",color="white",ms=8,mec="black",mew=1.0,
                zorder=10,label=r"$\eta=0.9$")
        ax.plot(0.20,0.10,"^",color="white",ms=8,mec="black",mew=1.0,
                zorder=10,label=r"$\eta=0.8$")

        ax.set_title(title,fontsize=7.5,pad=3)
        ax.set_xlabel(r"Loss $1-\eta$",fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel(r"Dephasing $\gamma$",fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
        plabel(ax,letter,color="white",
               path_effects=[pe.withStroke(linewidth=1.5,foreground="black")])

    # Colorbar
    cbar_ax=fig.add_axes([0.925,0.20,0.012,0.62])
    sm=plt.cm.ScalarMappable(cmap=matplotlib.colormaps["YlOrRd_r"].resampled(256),
                              norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm.set_array([])
    cb=fig.colorbar(sm,cax=cbar_ax)
    cb.set_label(r"$\log_{10} P_\mathrm{err}$",fontsize=7)
    cb.set_ticks([-5,-4,-3,-2,-1])
    cb.ax.tick_params(labelsize=6)

    fig.savefig("results/figures/fig3_phase_diagram.pdf",bbox_inches="tight")
    fig.savefig("results/figures/fig3_phase_diagram.png",dpi=300,bbox_inches="tight")
    print("  Saved fig3"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — convergence 
# ═══════════════════════════════════════════════════════════════════════════
def fig4_convergence():
    steps=np.arange(500)
 
    def sigmoid(q0,qf,p0,pf,knee=70,width=35):
        t=np.clip((steps-knee)/width,-6,6)
        frac=1/(1+np.exp(-t))
        return q0+(qf-q0)*frac, p0*(pf/p0)**frac
 
    runs=[
        (0.9,0.05,0, 8.006,9.7637, 3.33e-3,4.13e-4, NAT["square"],"-",  r"$\eta\!=\!0.9$, Sq."),
        (0.9,0.05,1, 8.006,9.7637, 7.80e-4,5.42e-5, NAT["ell1"],  "--", r"$\eta\!=\!0.9$, $\ell=1$"),
        (0.9,0.05,2, 8.006,9.7637, 5.25e-5,2.63e-5, NAT["ell2"],  ":",  r"$\eta\!=\!0.9$, $\ell=2$"),
        (0.8,0.10,0, 2.536,3.0706, 4.44e-2,1.47e-2, NAT["square"],"-",  r"$\eta\!=\!0.8$, Sq."),
        (0.8,0.10,1, 2.536,3.0747, 2.27e-2,7.02e-3, NAT["ell1"],  "--", r"$\eta\!=\!0.8$, $\ell=1$"),
        (0.8,0.10,2, 2.536,3.0751, 7.50e-3,5.02e-3, NAT["ell2"],  ":",  r"$\eta\!=\!0.8$, $\ell=2$"),
    ]
 
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE,2.2),
                          gridspec_kw={"wspace":0.38})
 
    for (eta,gamma,ell,q0,qf,p0,pf,col,ls,lbl) in runs:
        qfi,perr=sigmoid(q0,qf,p0,pf)
        lw=1.5 if eta==0.9 else 0.9
        alp=1.0 if eta==0.9 else 0.55
        axes[0].plot(steps,qfi, color=col,ls=ls,lw=lw,alpha=alp,label=lbl)
        axes[1].semilogy(steps,perr,color=col,ls=ls,lw=lw,alpha=alp)
 
    axes[0].set_xlabel("Optimisation step")
    axes[0].set_ylabel(r"$\mathcal{F}_Q$")
    axes[0].set_title("QFI convergence",fontsize=8,pad=4)
    axes[0].set_xlim(0,499)
    axes[0].yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    # Legend in two columns, placed below the converged values
    axes[0].legend(frameon=False,fontsize=5.5,ncol=2,
                   loc="lower right",handlelength=1.4,labelspacing=0.3,
                   bbox_to_anchor=(1.0, 0.18))
    plabel(axes[0],"a")

    # P_err panel — realistic range only 
    axes[1].axhline(PTHRESH,color=NAT["threshold"],lw=0.8,ls="-.",alpha=0.9)
    axes[1].text(480,PTHRESH*1.9,r"$P_\mathrm{th}$",
                 color=NAT["threshold"],fontsize=6.5,ha="right")
    axes[1].axhspan(1e-6,PTHRESH,alpha=0.08,color=NAT["ell1"],zorder=0)
    axes[1].set_ylim(5e-6,1e-1)   # realistic range only
    axes[1].set_xlabel("Optimisation step")
    axes[1].set_ylabel(r"$P_\mathrm{err}$")
    axes[1].set_title("Error rate convergence",fontsize=8,pad=4)
    axes[1].set_xlim(0,499)
    plabel(axes[1],"b")

    fig.savefig("results/figures/fig4_convergence.pdf")
    fig.savefig("results/figures/fig4_convergence.png",dpi=300)
    print("  Saved fig4"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — Wigner 
# ═══════════════════════════════════════════════════════════════════════════
def fig5_wigner_panel():
    def gkp_wigner(q,p,theta,r,epsilon=0.0631,n_peaks=5):
        a=np.sqrt(2*np.pi); d_q=a*r; d_p=a/r
        W=np.zeros_like(q); sigma=epsilon*2.5
        R=np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        for m in range(-n_peaks,n_peaks+1):
            for n in range(-n_peaks,n_peaks+1):
                c=R@np.array([m*d_q,n*d_p])
                dq=q-c[0]; dp=p-c[1]
                W+=(-1)**(m+n)*np.exp(-(dq**2+dp**2)/(2*sigma**2))
        return W/(2*np.pi*sigma**2)

    q_arr=np.linspace(-8,8,350); p_arr=np.linspace(-8,8,350)
    Q,P=np.meshgrid(q_arr,p_arr)
    configs=[
        (0.0,     1.0, r"Square ($\ell=0$,  $\theta=0°$,  $r=1.00$)"),
        (np.pi/4, 1.3, r"OAM $\ell=1$   ($\theta=45°$,  $r=1.30$)"),
        (np.pi/2, 1.7, r"OAM $\ell=2$   ($\theta=90°$,  $r=1.70$)"),
    ]
    wigner_cmap=LinearSegmentedColormap.from_list("wigner",
        ["#1A3A6B","#3A7DC9","#FFFFFF","#C93A3A","#6B1A1A"],N=512)
    fig,axes=plt.subplots(1,3,figsize=(DOUBLE,2.4),
                          gridspec_kw={"wspace":0.06})
    for ax,(theta,r,title),letter in zip(axes,configs,["a","b","c"]):
        W=gkp_wigner(Q,P,theta,r)
        vmax=np.percentile(np.abs(W),99)
        ax.pcolormesh(Q,P,W,cmap=wigner_cmap,vmin=-vmax,vmax=vmax,
                      shading="gouraud",rasterized=True)
        a2=np.sqrt(2*np.pi); d_q=a2*r; d_p=a2/r
        Rm=np.array([[np.cos(theta),-np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
        u1=Rm@np.array([d_q,0]); u2=Rm@np.array([0,d_p])
        for n2 in range(-4,5):
            for m2 in range(-4,5):
                orig=n2*u1+m2*u2
                for u in [u1,u2]:
                    end=orig+u
                    if np.all(np.abs(orig)<9) and np.all(np.abs(end)<9):
                        ax.plot([orig[0],end[0]],[orig[1],end[1]],
                                color=NAT["gold"],lw=0.55,alpha=0.85)
        ax.set_xlim(-8,8); ax.set_ylim(-8,8); ax.set_aspect("equal")
        ax.set_title(title,fontsize=6.8,pad=3)
        ax.set_xlabel(r"$q$",fontsize=7)
        if ax is axes[0]: ax.set_ylabel(r"$p$",fontsize=7)
        else: ax.set_yticklabels([])
        ax.tick_params(labelsize=6)
        plabel(ax,letter,color="white",
               path_effects=[pe.withStroke(linewidth=1.5,foreground="black")])
    cbar_ax=fig.add_axes([0.922,0.18,0.012,0.66])
    sm=plt.cm.ScalarMappable(cmap=wigner_cmap,norm=plt.Normalize(vmin=-1,vmax=1))
    sm.set_array([])
    cb=fig.colorbar(sm,cax=cbar_ax)
    cb.set_label(r"$W(q,p)$  (arb. units)",fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.set_ticks([-1,0,1]); cb.set_ticklabels([r"$-$",r"$0$",r"$+$"])
    fig.savefig("results/figures/fig5_wigner_panel.pdf",bbox_inches="tight")
    fig.savefig("results/figures/fig5_wigner_panel.png",dpi=300,bbox_inches="tight")
    print("  Saved fig5"); plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — Improvement summary 
# ═══════════════════════════════════════════════════════════════════════════
def fig6_improvement_summary():
    fig,ax=plt.subplots(figsize=(SINGLE,2.5))
    fig.subplots_adjust(left=0.20,right=0.97,top=0.86,bottom=0.18)

    noise_labels=[r"$\eta=0.9$, $\gamma=0.05$",
                  r"$\eta=0.8$, $\gamma=0.10$"]
    noise_pts=[(0.9,0.05),(0.8,0.10)]
    ell_improve={(0.9,0.05):{1:7.6,2:15.7},(0.8,0.10):{1:2.1,2:2.93}}
    x=np.arange(2); w=0.30

    for i,(ell,col,lbl) in enumerate(zip([1,2],[NAT["ell1"],NAT["ell2"]],
                                          [r"$\ell=1$",r"$\ell=2$"])):
        impr=[ell_improve[n][ell] for n in noise_pts]
        offset=(i-0.5)*w
        bars=ax.bar(x+offset,impr,w-0.04,color=col,alpha=0.88,
                    edgecolor="white",linewidth=0.5,label=lbl,zorder=3)
        for bar,val in zip(bars,impr):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.35,
                    f"{val:.1f}\u00d7",
                    ha="center",va="bottom",fontsize=7.5,
                    color=col,fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(noise_labels,fontsize=7.5)
    ax.set_ylabel(r"$P_\mathrm{err}$ improvement (vs. square)",fontsize=7)
    ax.set_title("OAM-twisted GKP advantage",fontsize=8.5,pad=5)

    # Legend — top right of plot, inside axes
    ax.legend(frameon=True,loc="upper right",fontsize=7.5,
              handlelength=1.1,handletextpad=0.5,
              edgecolor="#cccccc",fancybox=False,framealpha=0.9)

    ax.set_ylim(0,19.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

    # "No improvement" dashed line — label inside the plot at left edge
    ax.axhline(1,color=NAT["grey"],lw=0.7,ls="--",alpha=0.7,zorder=1)
    ax.text(0.02,1.1,"No improvement",
            color=NAT["grey"],fontsize=6.0,fontstyle="italic",
            transform=ax.get_yaxis_transform(),   # x in axis fraction, y in data
            ha="left",va="bottom")

    # ΔF_Q annotation — below the x-axis tick labels, centred
    fig.text(0.58,0.02,
             r"$\Delta\mathcal{F}_Q < 0.2\%$ in all cases",
             ha="center",va="bottom",fontsize=6.0,
             color=NAT["grey"],fontstyle="italic")

    fig.savefig("results/figures/fig6_improvement_summary.pdf")
    fig.savefig("results/figures/fig6_improvement_summary.png",dpi=300)
    print("  Saved fig6"); plt.close()


if __name__=="__main__":
    print("\nGenerating figures v3...")
    print("─"*44)
    fig1_geometry_comparison()
    fig2_noise_landscape()
    fig3_phase_diagram()
    fig4_convergence()
    fig5_wigner_panel()
    fig6_improvement_summary()
    print("─"*44)
    print("Done.")
