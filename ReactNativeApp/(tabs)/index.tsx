// app/(tabs)/index.tsx


import React, { useState, useRef } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity, ScrollView,
  Image, ActivityIndicator, Alert, Animated, Dimensions, Platform,
} from "react-native";
import { useAudioRecorder, RecordingPresets, AudioModule } from "expo-audio";
import * as DocumentPicker from "expo-document-picker";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { signOut } from "firebase/auth";
import { auth, db } from "@/config/firebase";
import { router } from "expo-router";


const API_URL = "https://heart-pcg-api-1058294664437.us-central1.run.app";

interface ModelVote { class: string; confidence: number; }
interface PredictionResult {
  predicted_class:     string;
  predicted_label:     string;
  confidence:          number;
  decided_by:          string;
  bpm:                 number | null;
  beats_detected:      number;
  all_probabilities:   Record<string, number>;
  model_votes:         Record<string, ModelVote>;
  waveform_image:      string;
  saliency_waveform:   string;
  gradcam_spectrogram: string;
}

const CLASS_COLORS: Record<string, string> = {
  AS: "#ff6b6b", MR: "#ffa94d", MS: "#ffd43b", MVP: "#a9e34b", N: "#40c057",
};
const CLASS_ICONS: Record<string, string> = {
  AS: "🫀", MR: "💔", MS: "🩺", MVP: "⚠️", N: "✅",
};

const BG = "#0a0f1a", CARD = "#111827", BORDER = "#1e2a3a",
      TEXT = "#e2e8f0", SUBTEXT = "#64748b", ACCENT = "#00d4aa";

export default function App() {
  const [phase,      setPhase]      = useState<"idle"|"recording"|"loading"|"result">("idle");
  const [result,     setResult]     = useState<PredictionResult | null>(null);
  const [recordSecs, setRecordSecs] = useState(0);
  const [activeTab,  setActiveTab]  = useState<"waveform"|"saliency"|"gradcam">("waveform");

  const pulseAnim = useRef(new Animated.Value(1)).current;
  const timerRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  const fadeAnim  = useRef(new Animated.Value(0)).current;

  const startPulse = () => Animated.loop(Animated.sequence([
    Animated.timing(pulseAnim, { toValue: 1.18, duration: 600, useNativeDriver: true }),
    Animated.timing(pulseAnim, { toValue: 1.0,  duration: 600, useNativeDriver: true }),
  ])).start();

  const stopPulse = () => { pulseAnim.stopAnimation(); pulseAnim.setValue(1); };
  const fadeIn    = () => { fadeAnim.setValue(0); Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }).start(); };

  const audioRecorder = useAudioRecorder(RecordingPresets.HIGH_QUALITY);

const startRecording = async () => {
  try {
    const { granted } = await AudioModule.requestRecordingPermissionsAsync();
    if (!granted) {
      Alert.alert("Permission required", "Microphone access is needed.");
      return;
    }
    await audioRecorder.record();
    setPhase("recording");
    setRecordSecs(0);
    startPulse();

    timerRef.current = setInterval(() => {
      setRecordSecs((s) => {
        if (s >= 5) { stopRecordingAuto(); return s; }
        return s + 1;
      });
    }, 1000);
  } catch (e) {
    Alert.alert("Error", "Could not start recording.");
  }
};

const stopRecordingAuto = async () => {
  if (timerRef.current) clearInterval(timerRef.current);
  stopPulse();
  try {
    await audioRecorder.stop();
    const uri = audioRecorder.uri;
    setPhase("loading");
    await sendAudio(uri!);
  } catch { setPhase("idle"); }
};

const stopRecording = async () => {
  if (timerRef.current) clearInterval(timerRef.current);
  stopPulse();
  try {
    await audioRecorder.stop();
    const uri = audioRecorder.uri;
    setPhase("loading");
    await sendAudio(uri!);
  } catch { setPhase("idle"); }
};
  const pickFile = async () => {
    try {
      const picked = await DocumentPicker.getDocumentAsync({ type: "audio/*", copyToCacheDirectory: true });
      if (picked.canceled) return;
      setPhase("loading");
      await sendAudio(picked.assets[0].uri);
    } catch { Alert.alert("Error", "Could not open file."); }
  };

  const sendAudio = async (uri: string) => {
    try {
      const formData = new FormData();
      formData.append("audio", { uri, type: "audio/wav", name: "recording.wav" } as any);

      formData.append("audio", { uri, type: "audio/m4a", name: "recording.m4a" } as any);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST", body: formData,
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (!response.ok) { const err = await response.json(); throw new Error(err.error || "Server error"); }

      const data: PredictionResult = await response.json();
      setResult(data); setPhase("result"); fadeIn();

      // Save to Firestore
      const user = auth.currentUser;
      if (user) {
        await addDoc(collection(db, "cases"), {
          doctorId:         user.uid,
          doctorName:       user.displayName ?? "Unknown",
          predictedClass:   data.predicted_class,
          predictedLabel:   data.predicted_label,
          confidence:       data.confidence,
          decidedBy:        data.decided_by,
          bpm:              data.bpm,
          beatsDetected:    data.beats_detected,
          allProbabilities: data.all_probabilities,
          modelVotes:       data.model_votes,
          doctorNote:       "",
          createdAt:        serverTimestamp(),
        });
      }
    } catch (e: any) {
      Alert.alert("Error", e.message || "Could not reach the server.");
      setPhase("idle");
    }
  };

  const reset = () => { setPhase("idle"); setResult(null); setRecordSecs(0); setActiveTab("waveform"); };
  const handleSignOut = async () => { await signOut(auth); router.replace("/login"); };

  return (
    <View style={styles.root}>
      <View style={styles.header}>
        <Text style={styles.headerIcon}>🫀</Text>
        <View style={{ flex: 1 }}>
          <Text style={styles.headerTitle}>CardioScope</Text>
          <Text style={styles.headerSub}>{auth.currentUser?.displayName ?? "Doctor"}</Text>
        </View>
        <TouchableOpacity style={styles.signOutBtn} onPress={handleSignOut}>
          <Text style={styles.signOutText}>Sign out</Text>
        </TouchableOpacity>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>

        {phase === "idle" && (
          <View style={styles.idleContainer}>
            <Text style={styles.instructions}>Record 5 seconds of heart sound or upload a WAV file to analyse.</Text>
            <TouchableOpacity style={styles.recordBtn} onPress={startRecording} activeOpacity={0.85}>
              <Text style={styles.recordBtnIcon}>🎙️</Text>
              <Text style={styles.recordBtnText}>Record Heart Sound</Text>
              <Text style={styles.recordBtnSub}>5 seconds · auto-stop</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.uploadBtn} onPress={pickFile} activeOpacity={0.85}>
              <Text style={styles.uploadBtnText}>📂  Upload WAV File</Text>
            </TouchableOpacity>
          </View>
        )}

        {phase === "recording" && (
          <View style={styles.recordingContainer}>
            <Animated.View style={[styles.pulseRing, { transform: [{ scale: pulseAnim }] }]} />
            <View style={styles.recordingInner}>
              <Text style={styles.recordingIcon}>🎙️</Text>
              <Text style={styles.recordingTimer}>{recordSecs}s / 5s</Text>
            </View>
            <Text style={styles.recordingLabel}>Recording heart sound…</Text>
            <TouchableOpacity style={styles.stopBtn} onPress={stopRecording}>
              <Text style={styles.stopBtnText}>⏹  Stop Early</Text>
            </TouchableOpacity>
          </View>
        )}

        {phase === "loading" && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={ACCENT} />
            <Text style={styles.loadingText}>Analysing heart sound…</Text>
            <Text style={styles.loadingSubText}>Running WST + MFCC + DenseNet121 + VGG16</Text>
          </View>
        )}

        {phase === "result" && result && (
          <Animated.View style={{ opacity: fadeAnim }}>

            <View style={[styles.diagnosisCard, { borderColor: CLASS_COLORS[result.predicted_class] ?? ACCENT }]}>
              <Text style={styles.diagnosisIcon}>{CLASS_ICONS[result.predicted_class] ?? "🫀"}</Text>
              <Text style={[styles.diagnosisClass, { color: CLASS_COLORS[result.predicted_class] ?? ACCENT }]}>
                {result.predicted_class}
              </Text>
              <Text style={styles.diagnosisLabel}>{result.predicted_label}</Text>
              <Text style={styles.diagnosisConfidence}>{(result.confidence * 100).toFixed(1)}% confidence</Text>
              <Text style={styles.diagnosisDecidedBy}>Decided by {result.decided_by}</Text>
            </View>

            <View style={styles.statsRow}>
              {[
                { value: result.bpm !== null ? `${result.bpm}` : "—", label: "BPM" },
                { value: `${result.beats_detected}`, label: "Beats" },
                { value: `${(result.confidence * 100).toFixed(0)}%`, label: "Confidence" },
              ].map((s) => (
                <View key={s.label} style={styles.statCard}>
                  <Text style={styles.statValue}>{s.value}</Text>
                  <Text style={styles.statLabel}>{s.label}</Text>
                </View>
              ))}
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Model Votes</Text>
              {Object.entries(result.model_votes).map(([name, vote]) => (
                <View key={name} style={styles.voteRow}>
                  <Text style={styles.voteName}>{name}</Text>
                  <Text style={[styles.voteClass, { color: CLASS_COLORS[vote.class] ?? ACCENT }]}>{vote.class}</Text>
                  <Text style={styles.voteConf}>{(vote.confidence * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Class Probabilities</Text>
              {Object.entries(result.all_probabilities).sort(([,a],[,b]) => b-a).map(([cls, prob]) => (
                <View key={cls} style={styles.probRow}>
                  <Text style={styles.probLabel}>{cls}</Text>
                  <View style={styles.probBarBg}>
                    <View style={[styles.probBarFill, { width: `${prob*100}%`, backgroundColor: CLASS_COLORS[cls] ?? ACCENT }]} />
                  </View>
                  <Text style={styles.probValue}>{(prob*100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Visualizations</Text>
              <View style={styles.tabRow}>
                {(["waveform","saliency","gradcam"] as const).map((tab) => (
                  <TouchableOpacity key={tab} style={[styles.tab, activeTab===tab && styles.tabActive]} onPress={() => setActiveTab(tab)}>
                    <Text style={[styles.tabText, activeTab===tab && styles.tabTextActive]}>
                      {tab==="waveform" ? "Waveform" : tab==="saliency" ? "Saliency" : "Grad-CAM"}
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
              <Text style={styles.vizDescription}>
                {activeTab==="waveform" ? "Preprocessed PCG signal with detected heartbeat markers."
                  : activeTab==="saliency" ? "Waveform colored by model attention — red regions drove the diagnosis."
                  : "Spectrogram with Grad-CAM heatmap showing which regions the model focused on."}
              </Text>
              <Image
                source={{ uri: `data:image/png;base64,${
                  activeTab==="waveform" ? result.waveform_image
                  : activeTab==="saliency" ? result.saliency_waveform
                  : result.gradcam_spectrogram}` }}
                style={styles.vizImage}
                resizeMode="contain"
              />
            </View>

            <View style={styles.savedNotice}>
              <Text style={styles.savedNoticeText}>✅ Case saved to your history</Text>
            </View>

            <TouchableOpacity style={styles.resetBtn} onPress={reset}>
              <Text style={styles.resetBtnText}>🔄  New Analysis</Text>
            </TouchableOpacity>

          </Animated.View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root:   { flex: 1, backgroundColor: BG },
  header: { flexDirection:"row", alignItems:"center", gap:12, paddingTop: Platform.OS==="ios"?56:44, paddingHorizontal:20, paddingBottom:16, borderBottomWidth:1, borderBottomColor:BORDER },
  headerIcon:  { fontSize: 32 },
  headerTitle: { fontSize: 20, fontWeight: "800", color: TEXT },
  headerSub:   { fontSize: 12, color: SUBTEXT },
  signOutBtn:  { borderWidth:1, borderColor:BORDER, borderRadius:8, paddingHorizontal:10, paddingVertical:5 },
  signOutText: { color: SUBTEXT, fontSize: 11 },
  scroll: { padding: 20, paddingBottom: 48 },

  idleContainer: { alignItems:"center", paddingTop:32, gap:16 },
  instructions:  { color:SUBTEXT, fontSize:14, textAlign:"center", lineHeight:21, marginBottom:8 },
  recordBtn:     { width:"100%", backgroundColor:ACCENT, borderRadius:16, paddingVertical:24, alignItems:"center", gap:4 },
  recordBtnIcon: { fontSize: 36 },
  recordBtnText: { color:"#0a0f1a", fontSize:18, fontWeight:"800" },
  recordBtnSub:  { color:"#0a4a3e", fontSize:12, fontWeight:"600" },
  uploadBtn:     { width:"100%", borderWidth:1.5, borderColor:BORDER, borderRadius:16, paddingVertical:16, alignItems:"center" },
  uploadBtnText: { color:TEXT, fontSize:15, fontWeight:"600" },

  recordingContainer: { alignItems:"center", paddingTop:48, gap:20 },
  pulseRing:    { position:"absolute", top:28, width:160, height:160, borderRadius:80, backgroundColor:"#00d4aa22", borderWidth:2, borderColor:"#00d4aa55" },
  recordingInner: { width:140, height:140, borderRadius:70, backgroundColor:"#00d4aa18", borderWidth:2, borderColor:ACCENT, alignItems:"center", justifyContent:"center", gap:4 },
  recordingIcon:  { fontSize: 40 },
  recordingTimer: { color:ACCENT, fontSize:18, fontWeight:"800" },
  recordingLabel: { color:SUBTEXT, fontSize:14, marginTop:8 },
  stopBtn:     { borderWidth:1, borderColor:"#ff6b6b", borderRadius:12, paddingHorizontal:24, paddingVertical:10 },
  stopBtnText: { color:"#ff6b6b", fontSize:14, fontWeight:"600" },

  loadingContainer: { alignItems:"center", paddingTop:80, gap:16 },
  loadingText:      { color:TEXT,    fontSize:18, fontWeight:"700" },
  loadingSubText:   { color:SUBTEXT, fontSize:12, textAlign:"center" },

  diagnosisCard:       { backgroundColor:CARD, borderRadius:20, borderWidth:2, padding:24, alignItems:"center", gap:6, marginBottom:16 },
  diagnosisIcon:       { fontSize: 48 },
  diagnosisClass:      { fontSize:36, fontWeight:"900", letterSpacing:1 },
  diagnosisLabel:      { color:TEXT,    fontSize:18, fontWeight:"600" },
  diagnosisConfidence: { color:SUBTEXT, fontSize:14 },
  diagnosisDecidedBy:  { color:SUBTEXT, fontSize:12, marginTop:2 },

  statsRow: { flexDirection:"row", gap:10, marginBottom:16 },
  statCard: { flex:1, backgroundColor:CARD, borderRadius:14, borderWidth:1, borderColor:BORDER, paddingVertical:14, alignItems:"center", gap:4 },
  statValue:{ color:ACCENT,  fontSize:22, fontWeight:"800" },
  statLabel:{ color:SUBTEXT, fontSize:11 },

  section:      { backgroundColor:CARD, borderRadius:16, borderWidth:1, borderColor:BORDER, padding:16, marginBottom:16, gap:10 },
  sectionTitle: { color:TEXT, fontSize:14, fontWeight:"700", letterSpacing:0.5, marginBottom:4 },

  voteRow:  { flexDirection:"row", alignItems:"center", justifyContent:"space-between", paddingVertical:6, borderBottomWidth:1, borderBottomColor:BORDER },
  voteName: { color:SUBTEXT, fontSize:13, flex:1 },
  voteClass:{ fontSize:15, fontWeight:"700", flex:1, textAlign:"center" },
  voteConf: { color:TEXT,   fontSize:13, flex:1, textAlign:"right" },

  probRow:    { flexDirection:"row", alignItems:"center", gap:10 },
  probLabel:  { color:SUBTEXT, fontSize:12, fontWeight:"600", width:36 },
  probBarBg:  { flex:1, height:8, backgroundColor:"#1e2a3a", borderRadius:4, overflow:"hidden" },
  probBarFill:{ height:8, borderRadius:4 },
  probValue:  { color:TEXT, fontSize:12, width:44, textAlign:"right" },

  tabRow:        { flexDirection:"row", gap:8, marginBottom:4 },
  tab:           { flex:1, paddingVertical:8, borderRadius:10, backgroundColor:"#1e2a3a", alignItems:"center" },
  tabActive:     { backgroundColor:ACCENT },
  tabText:       { color:SUBTEXT, fontSize:12, fontWeight:"600" },
  tabTextActive: { color:"#0a0f1a" },
  vizDescription:{ color:SUBTEXT, fontSize:11, lineHeight:16, marginBottom:4 },
  vizImage:      { width:"100%", height:200, borderRadius:10, backgroundColor:"#0d1117" },

  savedNotice:     { backgroundColor:"#0d2818", borderRadius:10, borderWidth:1, borderColor:"#40c057", padding:10, alignItems:"center", marginBottom:12 },
  savedNoticeText: { color:"#40c057", fontSize:13, fontWeight:"600" },

  resetBtn:     { borderWidth:1.5, borderColor:ACCENT, borderRadius:14, paddingVertical:14, alignItems:"center", marginTop:4 },
  resetBtnText: { color:ACCENT, fontSize:15, fontWeight:"700" },
});