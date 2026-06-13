// app/(tabs)/_layout.tsx

import { Tabs } from "expo-router";
import { Platform, Text } from "react-native";

//New
import { useSafeAreaInsets } from "react-native-safe-area-context";

export default function TabLayout() {

  //New
  const insets = useSafeAreaInsets();

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#0a0f1a",
          borderTopColor: "#1e2a3a",
          borderTopWidth: 1,
          //paddingBottom: Platform.OS === "ios" ? 24 : 25,
          //paddingTop: 8,
          //height: Platform.OS === "ios" ? 88 : 72,

          
          //New
          height: 52 + insets.bottom,
          paddingBottom: insets.bottom,
          paddingTop: 8,
        },
        
        tabBarActiveTintColor:   "#00d4aa",
        tabBarInactiveTintColor: "#64748b",
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: "600",
        },
      }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: "Analyse",
          tabBarIcon: () => <Text style={{ fontSize: 20 }}>🫀</Text>,
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          title: "History",
          tabBarIcon: () => <Text style={{ fontSize: 20 }}>📋</Text>,
        }}
      />
    </Tabs>
  );
}