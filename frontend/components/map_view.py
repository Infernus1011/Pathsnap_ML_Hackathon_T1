import folium
import streamlit as st

from streamlit_folium import st_folium


def display_map(
    gps_points=None,
    reference_route=None,
    predicted_route=None
):

    st.subheader("Geographical Route Visualization")

    gps_points = gps_points or []
    reference_route = reference_route or []
    predicted_route = predicted_route or []

    if not gps_points and not reference_route and not predicted_route:
        st.info(
            "Route geometry will appear after geographical "
            "data is integrated into the API response."
        )
        return

    all_points = (
        gps_points + reference_route + predicted_route
    )

    if not all_points:
        return

    center_lat = sum(p[0] for p in all_points) / len(all_points)
    center_lon = sum(p[1] for p in all_points) / len(all_points)

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15
    )

    if gps_points:

        folium.PolyLine(
            gps_points,
            color="blue",
            weight=3,
            tooltip="Original GPS"
        ).add_to(map_object)

        for point in gps_points:

            folium.CircleMarker(
                location=point,
                radius=3,
                color="blue",
                fill=True
            ).add_to(map_object)

    if reference_route:

        folium.PolyLine(
            reference_route,
            color="green",
            weight=5,
            tooltip="Reference Route"
        ).add_to(map_object)

    if predicted_route:

        folium.PolyLine(
            predicted_route,
            color="red",
            weight=5,
            tooltip="Predicted Route"
        ).add_to(map_object)

    folium.LayerControl().add_to(map_object)

    st_folium(
        map_object,
        width=None,
        height=550
    )
