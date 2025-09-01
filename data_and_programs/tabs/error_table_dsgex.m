%% Calculate Root Mean Squared Error of DSGE Model

clear;
year_=2010;
max_year=2019;
number_years=10;
number_quarters=4*number_years;
quarters_num=datenum(year_,4:3:number_years*12+1,0);
horizons=[1 2 4 8 12];
number_horizons=length(horizons);
number_variables=4;

forecast=NaN(number_quarters,number_horizons,number_variables);
actual=NaN(number_quarters,number_horizons,number_variables);

% Load data
load('../data/data/1996.mat');

% Calculate gdp deflator growth from levels
data.gdp_deflator_growth_quarterly(2:end)=exp(diff(log(data.nominal_gdp_quarterly./data.real_gdp_quarterly)))-1;

for i=1:number_quarters-1
    quarter_num=quarters_num(i);
    
    % Load data
    load(['../data/dsgex/',num2str(year(datetime(datestr(quarters_num(i))))),'Q',num2str(quarter(datetime(datestr(quarters_num(i))))),'.mat']);
    
    for j=1:number_horizons
        horizon=horizons(j);
        forecast_quarter_num=datenum(datetime(year(datetime(datestr(quarter_num))),month(datetime(datestr(quarter_num)))+(3*horizon)+1,0));
        
        if forecast_quarter_num>datenum(datetime(max_year,12,31))
            break
        end
        
        actual(i,j,:)=[log(data.real_gdp_quarterly(data.quarters_num==forecast_quarter_num)) ...
            log(1+data.gdp_deflator_growth_quarterly(data.quarters_num==forecast_quarter_num)) ...
            log(data.real_household_consumption_quarterly(data.quarters_num==forecast_quarter_num)) ...
            log(data.real_fixed_capitalformation_quarterly(data.quarters_num==forecast_quarter_num)) ...
            ];
        
        forecast(i,j,:)=[log(mean(dsgex.real_gdp_quarterly(dsgex.quarters_num==forecast_quarter_num,:),2)) ...
            log(1+mean(dsgex.gdp_deflator_growth_quarterly(dsgex.quarters_num==forecast_quarter_num,:),2)) ...
            log(mean(dsgex.real_household_consumption_quarterly(dsgex.quarters_num==forecast_quarter_num,:),2)) ...
            log(mean(dsgex.real_fixed_capitalformation_quarterly(dsgex.quarters_num==forecast_quarter_num,:),2)) ...
            ];
    end
end

save('forecast_dsgex','forecast','actual');
rmse_dsgex=100*squeeze(sqrt(nanmean((forecast-actual).^2)));
bias_dsgex=squeeze(nanmean(forecast-actual));
error_dsgex=forecast-actual;

load('forecast_arx','forecast','actual');
rmse_arx=100*squeeze(sqrt(nanmean((forecast-actual).^2)));
error_arx=forecast-actual;

% generate LaTex code error table
input.data=cellstr(string(round(100*(rmse_arx-rmse_dsgex)./rmse_arx,1)));

for j=1:length(horizons)
    h=horizons(j);
    for l=1:number_variables
        [~,p_value]=dmtest_modified(rmmissing(error_arx(:,j,l)),rmmissing(error_dsgex(:,j,l)),h);
        input.data{j,l}=[input.data{j,l},' (',num2str(round(p_value,2),'%.2f'),stars(p_value),')'];
    end
end

% Set row labels (use empty string for no label):
input.tableRowLabels = {'1q','2q','4q','8q','12q'};

% Formatting-string to set the precision of the table values:
input.dataFormat = {'%.2f'};

% Column alignment in Latex table ('l'=left-justified, 'c'=centered,'r'=right-justified):
input.tableColumnAlignment = 'r';

% Switch table borders on/off:
input.tableBorders = 0;

% Switch table booktabs on/off:
input.booktabs = 0;

% Switch to generate a complete LaTex document or just a table:
input.makeCompleteLatexDocument = 0;

latex = latexTableContent(input);

% save LaTex code as file
fid=fopen('rmse_dsgex.tex','w');
[nrows,ncols] = size(latex);
for row = 1:nrows
    fprintf(fid,'%s\n',latex{row,:});
end
fclose(fid);

% generate LaTex code bias table
input.data=cellstr(string(round(bias_dsgex,4)));

for j=1:length(horizons)
    h=horizons(j);
    for l=1:number_variables
        [~,~,p_value]=mztest(rmmissing(actual(:,j,l)),rmmissing(error_dsgex(:,j,l)+actual(:,j,l)));
        input.data{j,l}=[input.data{j,l},' (',num2str(round(p_value,2),'%.2f'),stars(p_value),')'];
    end
end

% Set row labels (use empty string for no label):
input.tableRowLabels = {'1q','2q','4q','8q','12q'};

% Formatting-string to set the precision of the table values:
input.dataFormat = {'%.2f'};

% Column alignment in Latex table ('l'=left-justified, 'c'=centered,'r'=right-justified):
input.tableColumnAlignment = 'r';

% Switch table borders on/off:
input.tableBorders = 0;

% Switch table booktabs on/off:
input.booktabs = 0;

% Switch to generate a complete LaTex document or just a table:
input.makeCompleteLatexDocument = 0;

latex = latexTableContent(input);

% save LaTex code as file
fid=fopen('bias_dsgex.tex','w');
[nrows,ncols] = size(latex);
for row = 1:nrows
    fprintf(fid,'%s\n',latex{row,:});
end
fclose(fid);
