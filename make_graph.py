import matplotlib.pyplot as plt
import csv
import os

# ==========================================
# 1. ファイルの指定とデータの読み込み
# ==========================================
csv_file_path = 'logs/evolution_metrics.csv'  # ★実際のCSVファイル名に書き換えてください

generations = []
cooperation_rates = []
unique_memes = []

if not os.path.exists(csv_file_path):
    print(f"エラー: '{csv_file_path}' が見つかりません。")
else:
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # ヘッダーをスキップ（もし1行目が文字なら）
        next(reader, None) 
        
        for row in reader:
            if not row or len(row) < 5:
                continue
            try:
                # Generation, Avg_Payoff, Cooperation_Rate, Mutants_Count, Unique_Memes
                gen = int(row[0])
                coop_rate = float(row[2])  # 3列目: 協力率
                memes = int(row[4])        # 5列目: ユニークな戦略数
                
                generations.append(gen)
                cooperation_rates.append(coop_rate)
                unique_memes.append(memes)
            except ValueError:
                continue

    if not generations:
        print("エラー: データを読み取れませんでした。")
    else:
        # ==========================================
        # 2. グラフの描画（協力率の推移）
        # ==========================================
        plt.figure(figsize=(10, 6))

        # 50%ラインを先に描画（データの下に配置するため）
        plt.axhline(y=50, color='#333333', linestyle='--', linewidth=1.2, alpha=0.6, label='50% Threshold')

        # 協力率をパーセント表記（0〜100%）に変換してプロット
        coop_percentages = [rate * 100 for rate in cooperation_rates]
        
        plt.plot(generations, coop_percentages, color='#2ca02c', linewidth=2.0, label='Cooperation Rate (%)')

        # グラフの装飾
        plt.title('Evolution of Cooperation in Spatial Prisoner\'s Dilemma (LLM Agents)', fontsize=14, fontweight='bold')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Cooperation Rate (%)', fontsize=12)
        plt.ylim(0, 100)  # Y軸は0%〜100%に固定
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.legend(loc='lower right')
        
        # 保存
        output_filename = 'cooperation_transition.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"成功！ 論文品質のグラフが {output_filename} として保存されました。")