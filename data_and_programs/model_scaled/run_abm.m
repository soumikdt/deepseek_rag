for year=2010:2019
    for quarter=1:4
        for predictors=0:1
            [nominal_gdp,real_gdp,nominal_gva,real_gva,nominal_household_consumption,real_household_consumption,nominal_government_consumption,real_government_consumption,nominal_capitalformation,real_capitalformation,nominal_fixed_capitalformation,real_fixed_capitalformation,nominal_fixed_capitalformation_dwellings,real_fixed_capitalformation_dwellings,nominal_exports,real_exports,nominal_imports,real_imports,operating_surplus,compensation_employees,wages,taxes_production,nominal_sector_gva,real_sector_gva,euribor,gdp_deflator_growth_ea,real_gdp_ea]=simulate_abm_mc(year,quarter,predictors);
            save([num2str(predictors),'_',num2str(year),'Q',num2str(quarter),'.mat'],'nominal_gdp', 'real_gdp', 'nominal_gva', 'real_gva', 'nominal_household_consumption', 'real_household_consumption', 'nominal_government_consumption', 'real_government_consumption', 'nominal_capitalformation', 'real_capitalformation', 'nominal_fixed_capitalformation', 'real_fixed_capitalformation', 'nominal_fixed_capitalformation_dwellings', 'real_fixed_capitalformation_dwellings', 'nominal_exports', 'real_exports', 'nominal_imports', 'real_imports', 'operating_surplus', 'compensation_employees', 'wages', 'taxes_production', 'nominal_sector_gva', 'real_sector_gva', 'euribor', 'gdp_deflator_growth_ea', 'real_gdp_ea');
        end
    end
end
