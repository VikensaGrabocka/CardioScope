// app/register.tsx


import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from "react-native";
import { createUserWithEmailAndPassword, updateProfile } from "firebase/auth";
import { doc, setDoc } from "firebase/firestore";
import { auth, db } from "@/config/firebase";
import { router } from "expo-router";

export default function Register() {
  const [name,     setName]     = useState("");
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [hospital, setHospital] = useState("");
  const [loading,  setLoading]  = useState(false);

  const register = async () => {
    if (!name || !email || !password) {
      Alert.alert("Error", "Please fill in all required fields.");
      return;
    }
    if (password.length < 6) {
      Alert.alert("Error", "Password must be at least 6 characters.");
      return;
    }
    setLoading(true);
    try {
      const { user } = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(user, { displayName: name });

      // Save doctor profile to Firestore
      await setDoc(doc(db, "doctors", user.uid), {
        name,
        email,
        hospital,
        createdAt: new Date().toISOString(),
      });

      router.replace("/(tabs)");
    } catch (e: any) {
      Alert.alert("Registration failed", e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: "#0a0f1a" }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView contentContainerStyle={styles.root}>
        <Text style={styles.icon}>🫀</Text>
        <Text style={styles.title}>Create Account</Text>
        <Text style={styles.subtitle}>Register as a medical professional</Text>

        <TextInput
          style={styles.input}
          placeholder="Full Name *"
          placeholderTextColor="#64748b"
          value={name}
          onChangeText={setName}
        />
        <TextInput
          style={styles.input}
          placeholder="Email *"
          placeholderTextColor="#64748b"
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        <TextInput
          style={styles.input}
          placeholder="Password * (min 6 characters)"
          placeholderTextColor="#64748b"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
        />
        <TextInput
          style={styles.input}
          placeholder="Hospital / Institution (optional)"
          placeholderTextColor="#64748b"
          value={hospital}
          onChangeText={setHospital}
        />

        <TouchableOpacity
          style={styles.btn}
          onPress={register}
          disabled={loading}
          activeOpacity={0.85}
        >
          <Text style={styles.btnText}>
            {loading ? "Creating account…" : "Create Account"}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity onPress={() => router.push("/login")}>
          <Text style={styles.link}>Already have an account? Sign in</Text>
        </TouchableOpacity>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: {
    flexGrow: 1,
    backgroundColor: "#0a0f1a",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
    gap: 14,
  },
  icon:     { fontSize: 56 },
  title:    { color: "#e2e8f0", fontSize: 28, fontWeight: "800" },
  subtitle: { color: "#64748b", fontSize: 14, marginBottom: 8 },
  input: {
    width: "100%",
    backgroundColor: "#111827",
    borderWidth: 1,
    borderColor: "#1e2a3a",
    borderRadius: 12,
    padding: 14,
    color: "#e2e8f0",
    fontSize: 15,
  },
  btn: {
    width: "100%",
    backgroundColor: "#00d4aa",
    borderRadius: 12,
    padding: 16,
    alignItems: "center",
  },
  btnText: { color: "#0a0f1a", fontSize: 16, fontWeight: "800" },
  link:    { color: "#00d4aa", fontSize: 13, marginTop: 4 },
});