/*------------------------------------------------------------------------------*
 * File Name:				 													*
 * Creation: 																	*
 * Purpose: OriginC Source C file												*
 * Copyright (c) ABCD Corp.	2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010		*
 * All Rights Reserved															*
 * 																				*
 * Modification Log:															*
 *------------------------------------------------------------------------------*/
 
////////////////////////////////////////////////////////////////////////////////////
// Including the system header file Origin.h should be sufficient for most Origin
// applications and is recommended. Origin.h includes many of the most common system
// header files and is automatically pre-compiled when Origin runs the first time.
// Programs including Origin.h subsequently compile much more quickly as long as
// the size and number of other included header files is minimized. All NAG header
// files are now included in Origin.h and no longer need be separately included.
//
// Right-click on the line below and select 'Open "Origin.h"' to open the Origin.h
// system header file.
#include <Origin.h>
////////////////////////////////////////////////////////////////////////////////////

//#pragma labtalk(0) // to disable OC functions for LT calling.

////////////////////////////////////////////////////////////////////////////////////
// Include your own header files here.
#include <stdio.h>
#include <string.h>

#include <..\originlab\NLFitSession.h>

////////////////////////////////////////////////////////////////////////////////////
// variables
#define Path_max_len 100
#define Name_max_len 20
#define HB_max_num 40
#define HB_view_num_max 50
#define HB_view_num_add 2
#define ShapeNet_max_num 3050
#define ShapeNet_view_num_max 50
#define ShapeNet_view_num_add 2
#define ShapeNet_batch_id 0

int i,j,k;

int HB_num;
char* HB_name[HB_max_num];
double HB_max_psnr[HB_max_num];

int ShapeNet_num;
char* ShapeNet_name[ShapeNet_max_num];
double ShapeNet_max_psnr[ShapeNet_max_num];

// functions
void Fit_ShapeNet( ) 
{ 
	char batch_id_str[3];
	sprintf(batch_id_str, "%d", ShapeNet_batch_id); 
	//printf("%s\n",batch_id_str);
	//init read buffer
	ShapeNet_num = 0;
	for(i=0;i<ShapeNet_max_num;i++)
		ShapeNet_name[i] = (char*) calloc(Name_max_len,sizeof(char));
	
	//read from file
	FILE *fp_names = fopen("D:/Data/NeRF_coverage/ShapeNet_names.txt", "r");
	while (fscanf(fp_names, "%s", ShapeNet_name[ShapeNet_num]) != EOF){
		//printf("%d %s\n",ShapeNet_num, ShapeNet_name[ShapeNet_num]);
		ShapeNet_num++;
	}
	//printf("%d\n",ShapeNet_num);
	fclose(fp_names);
	
	//use the workbook to put all the data 
	Worksheet wks = Project.ActiveLayer();	
	//set x col
	Column col_x(wks, 0);
	col_x.SetLongName("Views");
	vector<double>& vec_x = col_x.GetDataObject();
	for(i=3;i<=ShapeNet_view_num_max;i=i+ShapeNet_view_num_add){
		vec_x.Add((double)i);
	}
	//read y cols
	for(i=0;i<ShapeNet_num;i++){
		wks.AddCol();
		Column col_y(wks, i+1);
		col_y.SetLongName(ShapeNet_name[i]);
		vector<double>& vec_y = col_y.GetDataObject();
		char path_metrics[Path_max_len];
		strcpy(path_metrics,"D:/Data/NeRF_coverage/Coverage_images/ShapeNet_");
		strcat(path_metrics,batch_id_str);
		strcat(path_metrics,"_label/");
		strcat(path_metrics,ShapeNet_name[i]);
		for(j=3;j<=ShapeNet_view_num_max;j=j+ShapeNet_view_num_add){
			char path_metric[Path_max_len];
			strcpy(path_metric,path_metrics);
			strcat(path_metric,"/");
			char j_str[5];
			sprintf(j_str,"%d",j);
			strcat(path_metric,j_str);
			strcat(path_metric,".txt");
			FILE *fp_metric = fopen(path_metric, "r");
			char psnr[Path_max_len];
			fscanf(fp_metric, "PSNR\t%s", psnr);
			//printf("%f\n",atof(psnr));
			vec_y.Add(atof(psnr));
			fclose(fp_metric);
		}
		strcat(path_metrics,"/100.txt");
		FILE *fp_metric = fopen(path_metrics, "r");
		char max_psnr[Path_max_len];
		fscanf(fp_metric, "PSNR\t%s", max_psnr);
		//printf("%f\n",atof(max_psnr));
		ShapeNet_max_psnr[i] = atof(max_psnr);
		fclose(fp_metric);
	}
	
	//Fit curve for each object
	for(i=0;i<ShapeNet_num;i++){
		// Set Curve Function
		NLFitSession nlfSession;
		if ( !nlfSession.SetFunction("LognormalCDF") ){
			out_str("Fail to set function!");
			return;
		}
		// Set the dataset
		Column col_y(wks, i+1);
		vector<double>& vec_y = col_y.GetDataObject();
		if ( !nlfSession.SetData(vec_y, vec_x) ){
			out_str("Fail to set data for the dataset!");
			return;
		}    	
		// Parameter initialization
		if ( !nlfSession.ParamsInitValues() ){
			out_str("Fail to init parameters values!");
			return;
		}
		// Set the Fit Method
		nlfSession.SetFitMethod(FITMETH_ORTHOGONAL_DISTANCE_REGRESSION);
		if( !nlfSession.IsODRFit() ){
			out_str("Fail to set the fit method!");
			return;
		}
		// Do fit
		int nFitOutcome;
		nlfSession.Fit(&nFitOutcome);    
		bool converged = true;
		string strOutcome = nlfSession.GetFitOutCome(nFitOutcome);
		printf("%s:%s\n",ShapeNet_name[i],strOutcome);
		if( strncmp(strOutcome,"Fit converged.",14) != 0 && strncmp(strOutcome,"Fit did not converge. Maximum iteration setting of 100 was reached.",67) != 0){
			converged = false;
		}
		Column now_col_y(wks, i+1);
		vector<double>& now_vec_y = now_col_y.GetDataObject();
		for(j=0;j<now_vec_y.GetSize();j++)
			if( ShapeNet_max_psnr[i] < now_vec_y[j]) converged = false;
		// get ans and write to file
		char path_label[Path_max_len];
		strcpy(path_label,"D:/Data/NeRF_coverage/Coverage_images/ShapeNet_");
		strcat(path_label,batch_id_str);
		strcat(path_label,"_label/");
		strcat(path_label,ShapeNet_name[i]);
		strcat(path_label,"/label.txt");
		FILE *fp_label = fopen(path_label, "w+");
		if(!converged){
			fprintf(fp_label,"Converged 0\n");
		}
		else{
			fprintf(fp_label,"Converged 1\n");
		}
		vector<double> vec_x_test;
		for(j=3;j<=100;j=j++){
			vec_x_test.Add((double)j);
		}
		vector FitY(vec_x_test.GetSize());
		// Get fitted Y for the test points
		nlfSession.GetYFromX(vec_x_test, FitY, vec_x_test.GetSize(), 0);
		for(j=0;j<vec_x_test.GetSize();j++)
			fprintf(fp_label,"%d %f\n",j+3,FitY[j]);
		//label via gap to 100
		int gap;
		for(gap=0;gap<=10;gap++){
			fprintf(fp_label,"gap %d%% ",gap);
			for(j=0;j<vec_x_test.GetSize();j++)
				if(FitY[j]/ShapeNet_max_psnr[i] >= (1.0-0.01*gap)){
					fprintf(fp_label,"%d\n",j+3);
					break;
				}
			if(j==vec_x_test.GetSize())
				fprintf(fp_label,"-1\n");
			
		}
		//label via gradient
		double gradient;
		for(gradient=0.01;gradient<=0.20 + 1e-6;gradient+=0.01){
			fprintf(fp_label,"gradient %.2f ",gradient);
			for(j=1;j<vec_x_test.GetSize();j++)
				if(FitY[j] - FitY[j-1] <= gradient){
					fprintf(fp_label,"%d\n",j+3);
					break;
				}
			if(j==vec_x_test.GetSize())
				fprintf(fp_label,"-1\n");
		}
		fclose(fp_label);
	}
	
	//release buff
	for(i=0;i<ShapeNet_max_num;i++)
		free(ShapeNet_name[i]);
}


void Fit_HB( ) 
{ 
	//init read buffer
	HB_num = 0;
	for(i=0;i<HB_max_num;i++)
		HB_name[i] = (char*) calloc(Name_max_len,sizeof(char));
	
	//read from file
	FILE *fp_names = fopen("D:/Data/NeRF_coverage/HB_names.txt", "r");
	while (fscanf(fp_names, "%s", HB_name[HB_num]) != EOF){
		//printf("%d %s\n",HB_num, HB_name[HB_num]);
		HB_num++;
	}
	//printf("%d\n",HB_num);
	fclose(fp_names);
	
	//use the workbook to put all the data 
	Worksheet wks = Project.ActiveLayer();	
	//set x col
	Column col_x(wks, 0);
	col_x.SetLongName("Views");
	vector<double>& vec_x = col_x.GetDataObject();
	for(i=3;i<=HB_view_num_max;i=i+HB_view_num_add){
		if(i==13||i==17||i==31||i==41||i==47) continue;
		vec_x.Add((double)i);
	}
	//read y cols
	for(i=0;i<HB_num;i++){
		wks.AddCol();
		Column col_y(wks, i+1);
		col_y.SetLongName(HB_name[i]);
		vector<double>& vec_y = col_y.GetDataObject();
		char path_metrics[Path_max_len];
		strcpy(path_metrics,"D:/Data/NeRF_coverage/Coverage_images/");
		strcat(path_metrics,HB_name[i]);
		for(j=3;j<=HB_view_num_max;j=j+HB_view_num_add){
			if(j==13||j==17||j==41||j==31||j==47) continue;
			
			char path_metric[Path_max_len];
			strcpy(path_metric,path_metrics);
			strcat(path_metric,"/");
			char j_str[5];
			sprintf(j_str,"%d",j);
			strcat(path_metric,j_str);
			strcat(path_metric,".txt");
			FILE *fp_metric = fopen(path_metric, "r");
			char psnr[Path_max_len];
			fscanf(fp_metric, "PSNR\t%s", psnr);
			//printf("%d:%f\n",j,atof(psnr));
			vec_y.Add(atof(psnr));
			fclose(fp_metric);
		}
		strcat(path_metrics,"/100.txt");
		FILE *fp_metric = fopen(path_metrics, "r");
		char max_psnr[Path_max_len];
		fscanf(fp_metric, "PSNR\t%s", max_psnr);
		//printf("100:%f\n",atof(max_psnr));
		HB_max_psnr[i] = atof(max_psnr);
		fclose(fp_metric);
	}
	
	//Fit curve for each object
	for(i=0;i<HB_num;i++){
		// Set Curve Function
		NLFitSession nlfSession;
		if ( !nlfSession.SetFunction("LognormalCDF") ){
			out_str("Fail to set function!");
			return;
		}
		// Set the dataset
		Column col_y(wks, i+1);
		vector<double>& vec_y = col_y.GetDataObject();
		if ( !nlfSession.SetData(vec_y, vec_x) ){
			out_str("Fail to set data for the dataset!");
			return;
		}    	
		// Parameter initialization
		if ( !nlfSession.ParamsInitValues() ){
			out_str("Fail to init parameters values!");
			return;
		}
		// Set the Fit Method
		nlfSession.SetFitMethod(FITMETH_ORTHOGONAL_DISTANCE_REGRESSION);
		if( !nlfSession.IsODRFit() ){
			out_str("Fail to set the fit method!");
			return;
		}
		// Do fit
		int nFitOutcome;
		nlfSession.Fit(&nFitOutcome);    
		bool converged = true;
		string strOutcome = nlfSession.GetFitOutCome(nFitOutcome);
		printf("%s:%s\n",HB_name[i],strOutcome);
		if( strncmp(strOutcome,"Fit converged.",14) != 0){
			converged = false;
		}
		// get ans and write to file
		char path_label[Path_max_len];
		strcpy(path_label,"D:/Data/NeRF_coverage/Coverage_images/");
		strcat(path_label,HB_name[i]);
		strcat(path_label,"/label.txt");
		FILE *fp_label = fopen(path_label, "w+");
		if(!converged){
			fprintf(fp_label,"Converged 0\n");
		}
		else{
			fprintf(fp_label,"Converged 1\n");
		}
		vector<double> vec_x_test;
		for(j=3;j<=100;j=j++){
			vec_x_test.Add((double)j);
		}
		vector FitY(vec_x_test.GetSize());
		// Get fitted Y for the test points
		nlfSession.GetYFromX(vec_x_test, FitY, vec_x_test.GetSize(), 0);
		for(j=0;j<vec_x_test.GetSize();j++)
			fprintf(fp_label,"%d %f\n",j+3,FitY[j]);
		//label via gap to 100
		int gap;
		for(gap=0;gap<=10;gap++){
			fprintf(fp_label,"gap %d%% ",gap);
			for(j=0;j<vec_x_test.GetSize();j++)
				if(FitY[j]/HB_max_psnr[i] >= (1.0-0.01*gap)){
					fprintf(fp_label,"%d\n",j+3);
					break;
				}
			if(j==vec_x_test.GetSize())
				fprintf(fp_label,"-1\n");
			
		}
		//label via gradient
		double gradient;
		for(gradient=0.01;gradient<=0.20 + 1e-6;gradient+=0.01){
			fprintf(fp_label,"gradient %.2f ",gradient);
			for(j=1;j<vec_x_test.GetSize();j++)
				if(FitY[j] - FitY[j-1] <= gradient){
					fprintf(fp_label,"%d\n",j+3);
					break;
				}
			if(j==vec_x_test.GetSize())
				fprintf(fp_label,"-1\n");
		}
		fclose(fp_label);
	}
	
	//release buff
	for(i=0;i<HB_max_num;i++)
		free(HB_name[i]);
}
