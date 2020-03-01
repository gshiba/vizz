setwd("~/Desktop/WHO/who2007_R")
wfawho2007<-read.table("wfawho2007.txt",header=T,sep="",skip=0)
hfawho2007<-read.table("hfawho2007.txt",header=T,sep="",skip=0)
bfawho2007<-read.table("bfawho2007.txt",header=T,sep="",skip=0)

dat1<-read.csv("survey_who2007.csv",header=T,sep=",",skip=0,na.strings="")[-c(930:933),]
source("who2007.r")

who2007(FileLab = "svy_who2007", FilePath = "", mydf = dat1, sex = sex, age = agemons, weight = weight, height = height, sw=sw, oedema=oedema)

