import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
def clean(df: pd.DataFrame):
    def tld(s):
        p= re.split('\.',s)
        return p[-1]
    df.content = df.content.str.len()
    df['tld'] = df['tld'].apply(tld)
    df.label = df.label.apply(lambda x: 1 if x == 'good' else 0)
    df.who_is = df.who_is.apply(lambda x: 1 if x == 'complete' else 0)
    df.https = df.https.apply(lambda x: 1 if x == 'yes' else 0)
    return df
filename = "../gdx3pkwp47-2/Webpages_Classification_train_data.csv"
chunksize = 10000

figNo = 1
def analysisOnBasisOfGeoIp(df: pd.DataFrame):
    global figNo
    fig = plt.figure(figsize=(20,10))
    fig.suptitle('Country Distribution: Malicious vs Benign URLs in top 10 countries')
    # URLS Country wise
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Country')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
    tmp = df.groupby("geo_loc").filter(lambda x:len(x)>300)
    sns.countplot(x = 'geo_loc', data=tmp, hue='label',
        order=tmp['geo_loc'].value_counts().index[:10], ax=ax)
    fig.savefig(f"Fig{figNo:02}: Country Univariate Analysis.svg")
    figNo+=1

def analysisOnBasisOfAttribs(df):
    global figNo
    def onBasisOfWhois():
        global figNo
        fig = plt.figure(figsize=(20,10))
        fig.suptitle('Whois Distribution: Malicious vs Benign URLs on basis of completion of Whois records')
        # URLS Whois wise
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Whois')
        ax.set_ylabel('Count (Log))')
        ax.set_yscale('log')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
        tmp = df.groupby("who_is").filter(lambda x:len(x)>300)
        sns.countplot(x = 'who_is', data=tmp, hue='label',
            order=tmp['who_is'].value_counts().index, ax=ax)
        fig.savefig(f"Fig{figNo:02}: Whois Univariate Analysis.svg")
        figNo+=1
        plt.close()

    def onBasisOfTLD():
        global figNo
        fig = plt.figure(figsize=(20,10))
        fig.suptitle('TLD Distribution: Malicious vs Benign URLs on basis of TLD')
        # URLS Country wise
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('TLD')
        ax.set_ylabel('Count (Log)')
        ax.set_yscale('log')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
        tmp = df.groupby("tld").filter(lambda x:len(x)>300)
        sns.countplot(x = 'tld', data=tmp, hue='label',
            order=tmp['tld'].value_counts().index, ax=ax)
        fig.savefig(f"Fig{figNo:02}: TLD Univariate Analysis.svg")
        figNo+=1
        plt.close()

    def onBasisOfScheme():
        global figNo
        fig = plt.figure(figsize=(20,10))
        fig.suptitle('Scheme Distribution: Malicious vs Benign URLs on basis of Scheme')
        # URLS Country wise
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('https')
        ax.set_ylabel('Count (Log)')
        ax.set_yscale('log')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right', fontsize=14)
        tmp = df.groupby("https").filter(lambda x:len(x)>300)
        sns.countplot(x = 'https', data=tmp, hue='label',
            order=tmp['https'].value_counts().index, ax=ax)
        fig.savefig(f"Fig{figNo:02}: Scheme Univariate Analysis.svg")
        figNo+=1
        plt.close()

    onBasisOfScheme()
    onBasisOfWhois()
    onBasisOfTLD()

def analysisOnBasisOfLen(df):
    global figNo
    df_bad=df.loc[df['label']=='bad']
    df_good=df.loc[df['label']==1]
    def onBasisOfURL():
        global figNo
        # Histogram of Url Length: Malicious Webpages
        fig = plt.figure(figsize =(10,10))
        fig.suptitle("Url Length Distributioins: Malicious vs Benign Webpages")
        fig.subplots_adjust(wspace=0.6,hspace=0.4)
        ax = fig.add_subplot(3,2,1)
        ax.set_xlabel("URL Length of Malicious Webpages")
        ax.set_ylabel("Frequency")
        ax.text(70, 1200, r'$\mu$='+str(round(df_bad['url_len'].mean(),2)), fontsize=12)
        ax.hist(df_bad['url_len'], color='red', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of url_len: Malicious Webpages
        ax1 = fig.add_subplot(3,2,2)
        ax1.set_xlabel("URL Length of Malicious Webpages")
        ax1.set_ylabel("Frequency")
        sns.kdeplot(df_bad['url_len'], ax=ax1, shade=True, color='red')
        # Histogram of url_len: Benign Webpages
        ax2 = fig.add_subplot(3,2,3)
        ax2.set_xlabel("URL Length of Benign Webpages")
        ax2.set_ylabel("Frequency")
        ax2.text(70, 100000, r'$\mu$='+str(round(df_good['url_len'].mean(),2)), fontsize=12)
        ax2.hist(df_good['url_len'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of url_len: Benign Webpages
        ax3 = fig.add_subplot(3,2,4)
        ax3.set_xlabel("URL Length of Benign Webpages")
        ax3.set_ylabel("Frequency")
        sns.kdeplot(df_good['url_len'], ax=ax3, shade=True, color='green')
        # Violin Plots of 'url_len'
        ax4 = fig.add_subplot(3,1,3)
        sns.violinplot(x="label", y="url_len", data=df, ax=ax4)
        ax4.set_xlabel("Violin Plot: Distribution of URL Length vs Labels",size = 12,alpha=0.8)
        ax4.set_ylabel("Lenght of URL",size = 12,alpha=0.8)
        #Saving the Figs
        figc = fig
        figc.savefig(f"Fig{figNo:02}: All Plots- URL Length Univariate Analysis.svg")
        figNo+=1
        extent = ax.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}: URL Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax1.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:URL Length Density Plot Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax2.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:URL Length Histogram Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax3.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:URL Length Density Plot Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax4.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:URL Length Violin Plot-Benign & Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        plt.close()

    def onBasisOfJS():
        global figNo
        fig = plt.figure(figsize =(10,10))
        fig.suptitle("JavaScript Length Distributioins: Malicious vs Benign Webpages")
        fig.subplots_adjust(wspace=0.6,hspace=0.4)
        ax = fig.add_subplot(3,2,1)
        ax.set_xlabel("JS Length of Malicious Webpages")
        ax.set_ylabel("Frequency")
        ax.text(70, 1200, r'$\mu$='+str(round(df_bad['js_len'].mean(),2)), fontsize=12)
        ax.hist(df_bad['js_len'], color='red', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of js_len: Malicious Webpages
        ax1 = fig.add_subplot(3,2,2)
        ax1.set_xlabel("JS Length of Malicious Webpages")
        ax1.set_ylabel("Frequency")
        sns.kdeplot(df_bad['js_len'],ax=ax1,shade=True,color='red')
        # Histogram of js_len: Benign Webpages
        ax2 = fig.add_subplot(3,2,3)
        ax2.set_xlabel("JS Length of Benign Webpages")
        ax2.set_ylabel("Frequency")
        ax2.text(-8, 86000, r'$\mu$='+str(round(df_good['js_len'].mean(),2)), fontsize=12)
        ax2.hist(df_good['js_len'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of js_len: Benign Webpages
        ax3 = fig.add_subplot(3,2,4)
        ax3.set_xlabel("JS Length of Benign Webpages")
        ax3.set_ylabel("Frequency")
        sns.kdeplot(df_good['js_len'], ax=ax3, shade=True, color='green')
        # Violin Plots of 'js_len'
        ax4 = fig.add_subplot(3,1,3)
        sns.violinplot(x="label", y="js_len", data=df, ax=ax4)
        ax4.set_xlabel("Violin Plot: Distribution of JS Length vs Labels",size = 12,alpha=0.8)
        ax4.set_ylabel("Lenght of JavaScript (KB)",size = 12,alpha=0.8)
        #Saving the Figs
        figc = fig
        figc.savefig(f"Fig{figNo:02}: All Plots- JS Length Univariate Analysis.svg")
        figNo+=1
        extent = ax.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}: JS Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax1.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Length Density Plot Malicious.svg",bbox_inches=extent.expanded(1.7, 1.5))
        figNo+=1
        extent = ax2.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Length Histogram Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax3.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Length Density Plot Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax4.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Length Violin Plot-Benign & Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        plt.close()
    
    def onBasisOfJSObfuscated():
        global figNo
        fig = plt.figure(figsize =(10,10))
        fig.suptitle("JavaScript Obfuscated Length Distributioins: Malicious vs Benign Webpages")
        fig.subplots_adjust(wspace=0.6,hspace=0.4)
        ax = fig.add_subplot(3,2,1)
        ax.set_xlabel("JS Obfuscated Length of Malicious Webpages")
        ax.set_ylabel("Frequency (Log)")
        plt.yscale('log', nonpositive='clip')
        ax.text(70, 1200, r'$\mu$='+str(round(df_bad['js_obf_len'].mean(),2)), fontsize=12)
        ax.hist(df_bad['js_obf_len'], color='red', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of js_obf_len: Malicious Webpages
        ax1 = fig.add_subplot(3,2,2)
        ax1.set_xlabel("JS Obfuscated Length of Malicious Webpages")
        ax1.set_ylabel("Frequency (Log)")
        plt.yscale('log', nonpositive='clip')
        sns.kdeplot(df_bad['js_obf_len'],ax=ax1,shade=True,color='red')
        # Histogram of js_obf_len: Benign Webpages
        ax2 = fig.add_subplot(3,2,3)
        ax2.set_xlabel("JS Obfuscated Length of Benign Webpages")
        ax2.set_ylabel("Frequency")
        ax2.text(-8, 86000, r'$\mu$='+str(round(df_good['js_obf_len'].mean(),2)), fontsize=12)
        ax2.hist(df_good['js_obf_len'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Histogram of js_obf_len: Malicious Webpages
        ax3 = fig.add_subplot(3,2,4)
        ax3.set_xlabel("JS Obfuscated Length of Malicious Webpages")
        ax3.set_ylabel("Frequency")
        ax3.text(-8, 86000, r'$\mu$='+str(round(df_bad['js_obf_len'].mean(),2)), fontsize=12)
        ax3.hist(df_bad['js_obf_len'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Violin Plots of 'js_obf_len'
        ax4 = fig.add_subplot(3,1,3)
        sns.violinplot(x="label", y="js_obf_len", data=df, ax=ax4)
        ax4.set_xlabel("Violin Plot: Distribution of JS Obfuscated Length vs Labels",size = 12,alpha=0.8)
        ax4.set_ylabel("Lenght of JavaScript (KB)",size = 12,alpha=0.8)
        #Saving the Figs
        figc = fig
        figc.savefig(f"Fig{figNo:02}: All Plots- JS Obfuscated Length Univariate Analysis.png")
        figNo+=1
        extent = ax.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}: JS Obfuscated Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax1.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Obfuscated Length Density Plot Malicious.svg",bbox_inches=extent.expanded(1.7, 1.5))
        figNo+=1
        extent = ax2.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Obfuscated Length Histogram Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax3.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Obfuscated Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax4.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:JS Obfuscated Length Violin Plot-Benign & Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        plt.close()

    def onBasisOfContentLength():
        global figNo
        fig = plt.figure(figsize =(10,10))
        fig.suptitle("Content Length Distributioins: Malicious vs Benign Webpages")
        fig.subplots_adjust(wspace=0.6,hspace=0.4)
        ax = fig.add_subplot(3,2,1)
        ax.set_xlabel("Content Length of Malicious Webpages")
        ax.text(70, 1200, r'$\mu$='+str(round(df_bad['content'].mean(),2)), fontsize=12)
        ax.hist(df_bad['content'], color='red', bins=15, edgecolor='black', linewidth=1)
        # Density Plot of content: Malicious Webpages
        ax1 = fig.add_subplot(3,2,2)
        ax1.set_xlabel("Content Length of Malicious Webpages")
        sns.kdeplot(df_bad['content'],ax=ax1,shade=True,color='red')
        # Histogram of content: Benign Webpages
        ax2 = fig.add_subplot(3,2,3)
        ax2.set_xlabel("Content Length of Benign Webpages")
        ax2.set_ylabel("Frequency")
        ax2.text(-8, 86000, r'$\mu$='+str(round(df_good['content'].mean(),2)), fontsize=12)
        ax2.hist(df_good['content'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Histogram of content: Malicious Webpages
        ax3 = fig.add_subplot(3,2,4)
        ax3.set_xlabel("Content Length of Malicious Webpages")
        ax3.set_ylabel("Frequency")
        ax3.text(-8, 86000, r'$\mu$='+str(round(df_bad['content'].mean(),2)), fontsize=12)
        ax3.hist(df_bad['content'], color='green', bins=15, edgecolor='black', linewidth=1)
        # Violin Plots of 'content'
        ax4 = fig.add_subplot(3,1,3)
        sns.violinplot(x="label", y="content", data=df, ax=ax4)
        ax4.set_xlabel("Violin Plot: Distribution of Content Length vs Labels",size = 12,alpha=0.8)
        ax4.set_ylabel("Lenght of Content (KB)",size = 12,alpha=0.8)
        #Saving the Figs
        figc = fig
        figc.savefig(f"Fig{figNo:02}: All Plots- Content Length Univariate Analysis.png")
        figNo+=1
        extent = ax.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}: Content Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax1.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:Content Length Density Plot Malicious.svg",bbox_inches=extent.expanded(1.7, 1.5))
        figNo+=1
        extent = ax2.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:Content Length Histogram Benign.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax3.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:Content Length Histogram Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        extent = ax4.get_window_extent().transformed(figc.dpi_scale_trans.inverted())
        figc.savefig(f"Fig{figNo:02}:Content Length Violin Plot-Benign & Malicious.svg",bbox_inches=extent.expanded(1.6, 1.5))
        figNo+=1
        plt.close()

    onBasisOfURL()
    onBasisOfJS()
    onBasisOfJSObfuscated()
    onBasisOfContentLength()

def trivariate(df):
    global figNo
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    title = fig.suptitle("3D Trivariate Analysis: 'url_len','js_len' & 'js_obf_len'")
    xs = df.iloc[:,]['js_len']
    ys = df.iloc[:,]['js_obf_len']
    zs = df.iloc[:,]['url_len']
    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w',color='purple')
    ax.set_xlabel('js_len')
    ax.set_ylabel('js_obf_len')
    ax.set_zlabel('url_len')
    fig.savefig(f"Fig{figNo:02}: 3D Scatter Trivariate Analysis.png") 
    figNo+=1

def parallelCoordinates(df):
    global figNo
    # Scaling attribute values to avoid few outiers
    cols = ['url_len','js_len','js_obf_len']
    subset_df = df.iloc[:10000,][cols]
    ss = StandardScaler()
    scaled_df = ss.fit_transform(subset_df)
    scaled_df = pd.DataFrame(scaled_df, columns=cols)
    final_df = pd.concat([scaled_df, df.iloc[:10000,]['label']], axis=1)
    final_df
    # plot parallel coordinates
    fig=plt.figure(figsize = (12,7))
    title = fig.suptitle("Parallel Coordinates Plot: 'url_len','js_len' & 'js_obf_len'")
    pc = parallel_coordinates(final_df, 'label', color=('#FFE888', '#FF9999'))
    fig.savefig(f"Fig{figNo:02}: Parallel Coordinates Plot-Trivariate Analysis.png")
    figNo+=1

def scatterPlot(df):
    global figNo
    cols = ['url_len', 'js_len', 'js_obf_len','label']
    pp = sns.pairplot(df[cols], hue='label', height=1.8, aspect=1.8,
    palette={"good": "green", "bad": "red"},
    plot_kws=dict(edgecolor="black", linewidth=0.5))
    fig = pp.fig
    fig.subplots_adjust(top=0.93, wspace=0.3)
    t = fig.suptitle('Numerical Attributes : Pairwise Plot for both Malicious & Benign Webpages', fontsize=14)
    fig.savefig(f"Fig{figNo:02}: Scatter Plot-Trivariate Analysis.png")
    figNo+=1

def analysisOnBasisOfLabel(df):
    global figNo
    fig = plt.figure(figsize = (12,4))
    #fig.suptitle("Plot of Malicious and Benign Webpages", fontsize=14)
    fig.subplots_adjust(top=0.85, wspace=0.3)
    #Bar Plot
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_xlabel("Class Labels")
    ax1.set_ylabel("Frequency")
    ax1.title.set_text('Bar Plot: Malicious & Benign Webpages')
    labels = df['label'].value_counts()
    w = (list(labels.index), list(labels.values))
    ax1.tick_params(axis='both', which='major')
    bar = ax1.bar(w[0], w[1], color=['green','red'], edgecolor='black', linewidth=1)
    #Stacked Plot
    ax2 = fig.add_subplot(1,2,2)
    ax2.title.set_text('Stack Plot: Malicious & Benign Webpages')
    df.assign(dummy = 1).groupby(['dummy','label']).size().groupby(level=0).apply(
    lambda x: 100 * x / x.sum()).to_frame().unstack().plot(kind='bar',stacked=True,legend=False,ax=ax2,color={'red','green'}, linewidth=0.50, ec='k')
    ax2.set_xlabel('Benign/Malicious Webpages')# or it'll show up as 'dummy'
    ax2.set_xticks([])# disable ticks in the x axis
    current_handles, _ = plt.gca().get_legend_handles_labels()#Fixing Legend
    reversed_handles = reversed(current_handles)
    correct_labels = reversed(['Malicious','Benign'])
    plt.legend(reversed_handles,correct_labels)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.savefig(f"Fig{figNo:02}: Analysis on basis of label.svg")
    figNo+=1
    plt.close()

if __name__ == "__main__":
    if False:
        df = pd.DataFrame()
        i = 0
        with pd.read_csv(filename, chunksize=chunksize, index_col=0) as reader:
            df = pd.concat([clean(chunk) for chunk in reader], ignore_index=True)
        df.to_csv("tmp.csv", index=False)
    else:
        df = pd.read_csv("tmp.csv")

    good, bad = df.label.value_counts()
    total = good+bad

    # Analysis of Malicious and Benign web pages



    print(f"Malicious Urls in sample = {bad} ({100*bad/total}%)")
    print(f"Benign Urls in sample = {good} ({100*good/total}%)")


    # analysisOnBasisOfLabel(df)
    # print("Label Analysis Done")
    # analysisOnBasisOfAttribs(df)
    # print("Attribute Analysis Done")
    # analysisOnBasisOfGeoIp(df)
    # print("Geo Location Analysis Done")
    # analysisOnBasisOfLen(df)
    # print("Length Analysis Done")
    # trivariate(df)
    # print("Trivariate Analysis Done")
    parallelCoordinates(df)
    print("Parallel Coordinate Analysis Done")
    # scatterPlot(df)
    # print("Scatter Plot Analysis Done")
