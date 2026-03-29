import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Load model
with open('model.pkl', 'rb') as f:
    model, feature_columns = pickle.load(f)

st.set_page_config(page_title="AI Real Estate Pro", layout="wide")

# ================= UI STYLE =================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #0f2027, #2c5364);
    color: #00ffcc;
}

.title {
    font-size: 50px;
    text-align: center;
    font-weight: bold;
    color: #00ffcc;
    text-shadow: 0 0 20px #00ffcc;
    margin-bottom: 20px;
}

.glass {
    background: rgba(0,255,204,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(0,255,204,0.3);
    box-shadow: 0 0 20px rgba(0,255,204,0.3);
    text-align: center;
}

.stButton > button {
    background: linear-gradient(45deg, #00ffcc, #00ccff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    width: 100%;
    padding: 15px;
    font-size: 18px;
}

[data-testid="stSidebar"] {
    background: #000000;
}

.metric-card {
    background: rgba(0,255,204,0.1);
    padding: 15px;
    border-radius: 15px;
    border: 1px solid rgba(0,255,204,0.4);
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚡ AI House Price Predictor</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("📥 Property Details")

city = st.sidebar.text_input("City", "Mumbai").title()
area = st.sidebar.number_input("Area (sq ft)", 100, 20000, 2000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 10, 2)
stories = st.sidebar.number_input("Stories", 1, 10, 2)
parking = st.sidebar.number_input("Parking Spaces", 0, 10, 1)

mainroad = st.sidebar.selectbox("Main Road", ['yes', 'no'])
guestroom = st.sidebar.selectbox("Guest Room", ['yes', 'no'])
basement = st.sidebar.selectbox("Basement", ['yes', 'no'])
airconditioning = st.sidebar.selectbox("Air Conditioning", ['yes', 'no'])
prefarea = st.sidebar.selectbox("Preferred Area", ['yes', 'no'])
furnishingstatus = st.sidebar.selectbox("Furnishing", ['furnished', 'semi-furnished', 'unfurnished'])

# ================= HELPER FUNCTIONS =================
tier_1 = ["Mumbai","Delhi","Bangalore","Hyderabad"]
tier_2 = ["Pune","Ahmedabad","Jaipur"]

def get_city_factor(city_name):
    if city_name in tier_1:
        return 5000
    elif city_name in tier_2:
        return 3000
    else:
        return 2000

def predict_price(area_val, bedrooms_val, bathrooms_val, stories_val, parking_val,
                 mainroad_val, guestroom_val, basement_val, ac_val, prefarea_val,
                 furnishing_val, city_val):
    
    input_data = {
        'area': area_val,
        'bedrooms': bedrooms_val,
        'bathrooms': bathrooms_val,
        'stories': stories_val,
        'parking': parking_val,
        'mainroad_yes': int(mainroad_val == 'yes'),
        'guestroom_yes': int(guestroom_val == 'yes'),
        'basement_yes': int(basement_val == 'yes'),
        'airconditioning_yes': int(ac_val == 'yes'),
        'prefarea_yes': int(prefarea_val == 'yes'),
        'furnishingstatus_semi-furnished': int(furnishing_val == 'semi-furnished'),
        'furnishingstatus_unfurnished': int(furnishing_val == 'unfurnished')
    }
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    base_price = model.predict(input_df)[0]
    city_factor = get_city_factor(city_val)
    final_price = base_price + (area_val * city_factor)
    
    return final_price, base_price, city_factor

# ================= PREDICTION =================
if st.sidebar.button("🚀 Predict Price"):

    final_price, base_price, city_factor = predict_price(
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, airconditioning, prefarea,
        furnishingstatus, city
    )

    # ================= TOP METRICS =================
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f'<div class="glass"><h3>💰 Price</h3><h2>₹ {round(final_price):,}</h2></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="glass"><h3>📐 Area</h3><h2>{area} sq ft</h2></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="glass"><h3>🛏 Bedrooms</h3><h2>{bedrooms}</h2></div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="glass"><h3>📍 City</h3><h2>{city}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ================= TABS =================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Insights", "🎯 Feature Impact", "📊 Price Breakdown", "🧾 Report"])

    # ================= TAB 1: INSIGHTS =================
    with tab1:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("📊 Smart Insights")
            price_per_sqft = round(final_price / area)
            st.metric("Price per Sq.ft", f"₹ {price_per_sqft:,}")
            st.progress(min(final_price / 30000000, 1.0))

            if final_price > 20000000:
                st.error("🏰 Ultra-Premium Property Zone")
            elif final_price > 10000000:
                st.warning("💎 Premium Property Zone")
            elif final_price > 5000000:
                st.info("🏠 Mid-Range Property")
            else:
                st.success("💰 Affordable Range")
        
        with col_b:
            st.subheader("🏆 Property Score")
            
            # Calculate property score based on features
            score = 0
            if area > 3000: score += 20
            elif area > 2000: score += 15
            elif area > 1000: score += 10
            
            score += bedrooms * 5
            score += bathrooms * 5
            if parking >= 2: score += 10
            if mainroad == 'yes': score += 10
            if guestroom == 'yes': score += 5
            if basement == 'yes': score += 5
            if airconditioning == 'yes': score += 10
            if prefarea == 'yes': score += 10
            if furnishingstatus == 'furnished': score += 10
            
            score = min(score, 100)
            
            st.metric("Overall Score", f"{score}/100")
            st.progress(score / 100)
            
            if score >= 80:
                st.success("⭐ Excellent Property!")
            elif score >= 60:
                st.info("👍 Good Property")
            else:
                st.warning("📈 Room for Improvement")

    # ================= TAB 2: INTERACTIVE FEATURE IMPACT =================
    with tab2:
        st.subheader("🎯 Feature Impact Analysis")
        st.write("See how changing each feature affects the price in real-time!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Area impact
            st.write("**Area Impact**")
            area_range = st.slider("Adjust Area (sq ft)", 500, 10000, area, key="area_slider")
            price_with_area, _, _ = predict_price(
                area_range, bedrooms, bathrooms, stories, parking,
                mainroad, guestroom, basement, airconditioning, prefarea,
                furnishingstatus, city
            )
            price_diff_area = price_with_area - final_price
            st.metric("Price Change", f"₹ {abs(round(price_diff_area)):,}", 
                     delta=f"₹ {round(price_diff_area):,}")
            
            # Bedrooms impact
            st.write("**Bedrooms Impact**")
            bedrooms_range = st.slider("Adjust Bedrooms", 1, 10, bedrooms, key="bed_slider")
            price_with_bed, _, _ = predict_price(
                area, bedrooms_range, bathrooms, stories, parking,
                mainroad, guestroom, basement, airconditioning, prefarea,
                furnishingstatus, city
            )
            price_diff_bed = price_with_bed - final_price
            st.metric("Price Change", f"₹ {abs(round(price_diff_bed)):,}", 
                     delta=f"₹ {round(price_diff_bed):,}")
        
        with col2:
            # Bathrooms impact
            st.write("**Bathrooms Impact**")
            bathrooms_range = st.slider("Adjust Bathrooms", 1, 10, bathrooms, key="bath_slider")
            price_with_bath, _, _ = predict_price(
                area, bedrooms, bathrooms_range, stories, parking,
                mainroad, guestroom, basement, airconditioning, prefarea,
                furnishingstatus, city
            )
            price_diff_bath = price_with_bath - final_price
            st.metric("Price Change", f"₹ {abs(round(price_diff_bath)):,}", 
                     delta=f"₹ {round(price_diff_bath):,}")
            
            # Parking impact
            st.write("**Parking Impact**")
            parking_range = st.slider("Adjust Parking Spaces", 0, 5, parking, key="park_slider")
            price_with_park, _, _ = predict_price(
                area, bedrooms, bathrooms, stories, parking_range,
                mainroad, guestroom, basement, airconditioning, prefarea,
                furnishingstatus, city
            )
            price_diff_park = price_with_park - final_price
            st.metric("Price Change", f"₹ {abs(round(price_diff_park)):,}", 
                     delta=f"₹ {round(price_diff_park):,}")
        
        # Feature comparison chart
        st.markdown("---")
        st.write("**Feature Impact Comparison**")
        
        features_impact = {
            'Feature': ['Area +1000 sqft', 'Bedrooms +1', 'Bathrooms +1', 'Parking +1'],
            'Price Impact': []
        }
        
        # Calculate impact of +1 for each feature
        p1, _, _ = predict_price(area+1000, bedrooms, bathrooms, stories, parking, mainroad, guestroom, basement, airconditioning, prefarea, furnishingstatus, city)
        features_impact['Price Impact'].append(round(p1 - final_price))
        
        p2, _, _ = predict_price(area, bedrooms+1, bathrooms, stories, parking, mainroad, guestroom, basement, airconditioning, prefarea, furnishingstatus, city)
        features_impact['Price Impact'].append(round(p2 - final_price))
        
        p3, _, _ = predict_price(area, bedrooms, bathrooms+1, stories, parking, mainroad, guestroom, basement, airconditioning, prefarea, furnishingstatus, city)
        features_impact['Price Impact'].append(round(p3 - final_price))
        
        p4, _, _ = predict_price(area, bedrooms, bathrooms, stories, parking+1, mainroad, guestroom, basement, airconditioning, prefarea, furnishingstatus, city)
        features_impact['Price Impact'].append(round(p4 - final_price))
        
        df_impact = pd.DataFrame(features_impact)
        
        fig_impact = px.bar(df_impact, x='Feature', y='Price Impact',
                           title='Impact of Adding 1 Unit of Each Feature',
                           color='Price Impact',
                           color_continuous_scale='Teal')
        fig_impact.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#00ffcc')
        )
        st.plotly_chart(fig_impact, use_container_width=True)

    # ================= TAB 3: PRICE BREAKDOWN =================
    with tab3:
        st.subheader("📊 Price Breakdown Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart showing base price vs city premium
            city_premium = area * city_factor
            
            breakdown_data = {
                'Component': ['Base Price (Model)', f'City Premium ({city})'],
                'Amount': [round(base_price), round(city_premium)]
            }
            df_breakdown = pd.DataFrame(breakdown_data)
            
            fig_pie = px.pie(df_breakdown, values='Amount', names='Component',
                            title='Price Components',
                            color_discrete_sequence=['#00ffcc', '#00ccff'])
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ffcc')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart showing price range comparison
            st.write("**City-wise Price Comparison**")
            
            city_comparison = []
            for test_city in ["Mumbai", "Delhi", "Bangalore", "Pune", "Nashik"]:
                test_price, _, _ = predict_price(
                    area, bedrooms, bathrooms, stories, parking,
                    mainroad, guestroom, basement, airconditioning, prefarea,
                    furnishingstatus, test_city
                )
                city_comparison.append({
                    'City': test_city,
                    'Estimated Price': round(test_price),
                    'Current': 'Yes' if test_city == city else 'No'
                })
            
            df_cities = pd.DataFrame(city_comparison)
            
            fig_cities = px.bar(df_cities, x='City', y='Estimated Price',
                               color='Current',
                               title='Same Property in Different Cities',
                               color_discrete_map={'Yes': '#00ffcc', 'No': '#00ccff'})
            fig_cities.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ffcc')
            )
            st.plotly_chart(fig_cities, use_container_width=True)
        
        # Feature value contribution
        st.markdown("---")
        st.write("**Feature Value Contribution**")
        
        feature_values = {
            'Feature': [],
            'Status': [],
            'Impact': []
        }
        
        # Binary features
        if mainroad == 'yes':
            feature_values['Feature'].append('Main Road')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+5%')
        
        if guestroom == 'yes':
            feature_values['Feature'].append('Guest Room')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+3%')
        
        if basement == 'yes':
            feature_values['Feature'].append('Basement')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+4%')
        
        if airconditioning == 'yes':
            feature_values['Feature'].append('Air Conditioning')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+6%')
        
        if prefarea == 'yes':
            feature_values['Feature'].append('Preferred Area')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+8%')
        
        if furnishingstatus == 'furnished':
            feature_values['Feature'].append('Furnished')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+10%')
        elif furnishingstatus == 'semi-furnished':
            feature_values['Feature'].append('Semi-Furnished')
            feature_values['Status'].append('✓')
            feature_values['Impact'].append('+5%')
        
        if feature_values['Feature']:
            df_features = pd.DataFrame(feature_values)
            st.dataframe(df_features, use_container_width=True)
        else:
            st.info("No premium features selected")

    # ================= TAB 4: REPORT =================
    with tab4:
        st.subheader("🧾 Detailed Property Report")
        
        report = f"""
╔════════════════════════════════════════════════════════════╗
║           AI House Price Predictor                         ║             
╚════════════════════════════════════════════════════════════╝

📍 LOCATION   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
City: {city}
City Tier: {'Tier 1' if city in tier_1 else 'Tier 2' if city in tier_2 else 'Tier 3'}
City Factor: ₹{city_factor:,} per sq ft

🏠 PROPERTY DETAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Area: {area} sq ft
Bedrooms: {bedrooms}
Bathrooms: {bathrooms}
Stories: {stories}
Parking Spaces: {parking}

✨ FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Main Road Access: {mainroad}
Guest Room: {guestroom}
Basement: {basement}
Air Conditioning: {airconditioning}
Preferred Area: {prefarea}
Furnishing Status: {furnishingstatus}

💰 PRICING ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base Price (Model): ₹{round(base_price):,}
City Premium: ₹{round(area * city_factor):,}
──────────────────────────────────────────────────────────
FINAL PREDICTED PRICE: ₹{round(final_price):,}
──────────────────────────────────────────────────────────
Price per Sq.ft: ₹{round(final_price/area):,}

📊 MARKET ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Property Category: {'Ultra-Premium' if final_price > 20000000 else 'Premium' if final_price > 10000000 else 'Mid-Range' if final_price > 5000000 else 'Affordable'}
Market Position: {'Top 10%' if final_price > 20000000 else 'Top 25%' if final_price > 10000000 else 'Middle 50%' if final_price > 5000000 else 'Budget Friendly'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by: AI Real Estate Pro
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        st.text(report)
        st.download_button("📥 Download Full Report", report, file_name=f"property_report_{city}_{area}sqft.txt")
        
        # Additional insights
        st.markdown("---")
        st.write("**💡 Investment Insights**")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info(f"**ROI Potential**: {'High' if final_price < 8000000 else 'Moderate'}")
            st.info(f"**Rental Yield**: Estimated {round(final_price * 0.00015):,}/month")
        
        with insights_col2:
            st.success(f"**Appreciation**: {round(final_price * 0.08):,}/year (8% avg)")
            st.success(f"**Market Liquidity**: {'High' if city in tier_1 else 'Moderate'}")

st.markdown("---")
st.caption("⚡ AI Real Estate Pro - Advanced ML Dashboard | Powered by Machine Learning")
