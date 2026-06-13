// app/login.tsx


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
} from "react-native";
import { signInWithEmailAndPassword } from "firebase/auth";
import { auth } from "@/config/firebase";
import { router } from "expo-router";

export default function Login() {
  const [email,    setEmail]    = useState("");
  const [password, setPassword] = useState("");
  const [loading,  setLoading]  = useState(false);

  const login = async () => {
    if (!email || !password) {
      Alert.alert("Error", "Please fill in all fields.");
      return;
    }
    setLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.replace("/(tabs)");
    } catch (e: any) {
      Alert.alert("Login failed", e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <Text style={styles.icon}>🫀</Text>
      <Text style={styles.title}>CardioScope</Text>
      <Text style={styles.subtitle}>Sign in to continue</Text>

      <TextInput
        style={styles.input}
        placeholder="Email"
        placeholderTextColor="#64748b"
        value={email}
        onChangeText={setEmail}
        keyboardType="email-address"
        autoCapitalize="none"
      />
      <TextInput
        style={styles.input}
        placeholder="Password"
        placeholderTextColor="#64748b"
        value={password}
        onChangeText={setPassword}
        secureTextEntry
      />

      <TouchableOpacity
        style={styles.btn}
        onPress={login}
        disabled={loading}
        activeOpacity={0.85}
      >
        <Text style={styles.btnText}>
          {loading ? "Signing in…" : "Sign In"}
        </Text>
      </TouchableOpacity>

      <TouchableOpacity onPress={() => router.push("/register")}>
        <Text style={styles.link}>Don't have an account? Register</Text>
      </TouchableOpacity>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
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